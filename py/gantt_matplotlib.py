import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pandas import Timestamp

#INPUT_FOLDER = "py/workflows_v03/CYBERSHAKE50 (volume x1)/"
#INPUT_FOLDER = "py/workflows_v03/LIGO50/"
#INPUT_FOLDER = "py/workflows_v03/GENOME50/"
INPUT_FOLDER = "py/workflows_v03/MONTAGE1000/"

def loadTasks(path = INPUT_FOLDER):
    tasks = pd.read_csv(path + "task_time_table.csv", delimiter=',')
    tasks["task_id"]=tasks["task_id"].values.astype('str')
    tasks['task'] = tasks['task_id'] + "/" + tasks['task_name']
    tasks.rename(columns={'start_time':'start', 'finish_time':'end'}, inplace=True)
    #tasks['start'] = tasks['start']*10  # cause some tasks have < 1 interval
    #tasks['end'] = tasks['end']*10
    #tasks['volume'] = tasks['volume']*10
    data_transfers = pd.read_csv(path + "transfer_size_table.csv", delimiter=',')
    data_transfers.rename(columns={'task_from':'source_task', 'task_to':'target_task', 'transfer_size': 'data_volume' }, inplace=True)

    tasks['index'] = tasks['task']
    tasks.set_index('index', inplace = True)

    tasks["task_type"] = 'Task'
    tasks["task_status"] = 'None'
    tasks["assigned_vm"] = pd.NA
    tasks["allocation_end"] = pd.NA

    # days between start and end of each task
    tasks['interval'] = tasks.end - tasks.start

    return tasks, data_transfers

def loadVMs(path = INPUT_FOLDER):
    vm_counter = 0
    #vm_types = pd.read_csv(path + "processor_table.csv", delimiter=',')
    vm_types = pd.read_csv("py/processor_table.csv", delimiter=',')
    vm_types.rename(columns={'performance':'perf', 'VM_type': "vm_type"}, inplace=True)
    vm_types['cost'] = vm_types['perf']
    vm_types['vm_start_time'] = 1
    vm_types['vm_stop_time'] = 1
    vms = pd.DataFrame()
    print("VM TYPES:")
    print(vm_types)

    return vm_types, vms

def calcTimingsForVM(vm_perf, tasks):
    tasks['calc_time'] = (tasks.volume / vm_perf)
    tasks['latest_start'] = (tasks.end - tasks.calc_time)
    tasks['earliest_finish'] = (tasks.start + tasks.calc_time)

def assignToLeader(vm_types, tasks, time, num):
    
    # first find leader and recalculate possible tasks timings
    leader_vm = vm_types.nlargest(1, 'perf').iloc[0]
    calcTimingsForVM(leader_vm['perf'], tasks)   
    
    # next 
    tasks.loc[tasks.task_status == 'None', "possible_start"] = np.maximum(tasks.start, time)
    ttasks = tasks[tasks.task_status == 'None']  # next available tasks to schedule

    ttasks.earliest_finish = ttasks.possible_start + tasks.calc_time
    EFT = ttasks.earliest_finish.min()
    
    idx = ttasks[ttasks.earliest_finish == EFT].index[0]
    tasks.loc[idx, ['task_status', 'batch', 'finish_time']] = 'Leader', 'Batch ' + str(num), EFT
    #tasks.loc[(ttasks.earliest_finish == EFT) & (tasks.task_status == 'None'), ['task_status', 'batch', 'finish_time']] = 'Leader', 'Batch ' + str(num), EFT
    tasks.loc[(ttasks.latest_start < EFT) & (tasks.task_status == 'None'), ['task_status', 'batch', 'finish_time']] = 'Batch', 'Batch ' + str(num), EFT
    #print(EFT)

    return EFT

def retrieveParalellBatchesFTL(vm_types, tasks, time):
    batch_num = 1
    tasks.loc[(tasks.task.str.contains('entry')) | (tasks.task.str.contains('finish')), 'task_status'] = 'IO'
    while('None' in tasks.task_status.values):
        time = assignToLeader(vm_types, tasks, time, batch_num)
        batch_num += 1

def checkTaskReady(task, tasks, data_transfers):
    related_tasks_from = data_transfers[data_transfers['target_task'] == task]['source_task'].tolist()
    not_assigned_tasks_from = tasks.loc[tasks.task.isin(related_tasks_from) & (tasks.batch == ""), 'task'].tolist()

    return len(not_assigned_tasks_from) == 0

def assignReadyTasks(vm_types, tasks, transfers, time, num):
    
    tasks.loc[tasks.task_status == 'None', "possible_start"] = np.maximum(tasks.start, time)

    batch_str = 'Batch ' + str(num)
    tasks['ready'] = tasks['task'].apply(lambda x: checkTaskReady(x, tasks, transfers))
    tasks.loc[(tasks.batch == '') & (tasks.ready), ['task_status', 'batch']] = 'Batch', batch_str
    tasks.loc[tasks.batch == batch_str, 'finish_time'] = tasks.loc[tasks.batch == batch_str, 'possible_start'] + tasks.loc[tasks.batch == batch_str, 'calc_time']
    min_finish_time = tasks.loc[tasks.batch == batch_str, 'end'].min()
    return min_finish_time
    


def retrieveParalellBatchesASAP(vm_types, tasks, transfers, time):
    batch_num = 0
    tasks['batch'] = ''
    #tasks.loc[(tasks.task.str.contains('entry')) | (tasks.task.str.contains('finish')), 'task_status'] = 'IO'
    tasks.loc[(tasks.task.str.contains('entry')), 'start'] = 0
    tasks.loc[(tasks.task.str.contains('entry')), 'end'] = 0
    tasks.loc[(tasks.task.str.contains('finish')), 'start'] = 400
    tasks.loc[(tasks.task.str.contains('finish')), 'end'] = 800

    leader_vm = vm_types.nlargest(1, 'perf').iloc[0]
    calcTimingsForVM(leader_vm['perf'], tasks)
    
    while('None' in tasks.task_status.values):
        time = assignReadyTasks(vm_types, tasks, transfers, time, batch_num)
        batch_num += 1



def countDataReuse(input_tasks, data_transfers):
    reuseCnt = 0
    tasks = input_tasks[~input_tasks.task.str.contains('entry|finish')]
    for index, row in tasks.iterrows():
        batch_num = row['batch'].split(' ')[1]
        previous_batch = 'Batch ' + str(int(batch_num) - 1)
        parent_tasks = data_transfers[data_transfers['target_task'] == row['task']]['source_task'].tolist()
        parent_tasks_in_previous_batch = tasks.loc[tasks.task.isin(parent_tasks) & (tasks.batch == previous_batch), 'task'].tolist()
        if len(parent_tasks_in_previous_batch) > 0:
            reuseCnt += 1
            
    return reuseCnt

def color(row):
    standard_colors = [
        
        "#00FF00",  # Green
        "#0000FF",  # Blue
        "#555500",  # Orange
        "#00FFFF",  # Cyan
        "#800000",  # Maroon
        "#800080",  # Purple
        
    ]
    print(row.batch)
    if row.task_status == 'Leader':
        return "#FF0000"  # Red
    else:
        return "#34D05C"  # Green #standard_colors[int(row.batch.split(' ')[1]) - 1]
def color_old(row):
    c_dict = {'Leader':'#E64646', 'CPU 2':'#E69646', 'Batch':'#34D05C', 'CPU 4':'#34D0C3', 'CPU 5':'#3475D0', 'None' : '#000000', 'IO' : '#000000'}
    if row['task_status'] in c_dict.keys():
        return c_dict[row['task_status']]
    else:
        return '#000000'

def runExperiment():

    tasks, data_transfers = loadTasks()
    vm_types, vms = loadVMs()


    #retrieveParalellBatchesFTL(vm_types, tasks, 0)
    retrieveParalellBatchesASAP(vm_types, tasks, data_transfers, 0)

    
    tasks = tasks[tasks.task_name.notnull()]
    tasks['finish_time'] = tasks['possible_start'] + tasks['calc_time']
    df = tasks.sort_values(['batch', 'finish_time', 'task_status', 'start'], ascending = [True, True, False, True])
    print("Tasks with possible data reuse: {}".format(countDataReuse(tasks, data_transfers)))

    df = df[~df.task.str.contains('entry|finish')]

    # create a column with the color for each department

    df['color'] = df.apply(color, axis=1)

    ##### PLOT #####
    fig, (ax, ax1) = plt.subplots(2, figsize=(16,6), gridspec_kw={'height_ratios':[100, 1]})

    # bars

    #ax.barh(df.Task, df.calc_days, left=df.LST, color=df.color, fill=False, hatch='---')
    ax.barh(df.task, df.interval, left=df.start, color=df.color, alpha=0.3)
    ax.barh(df.task, df.calc_time, left=df.possible_start, color=df.color)
    #ax.barh(df.Task, df.calc_days, left=df.start_num, color=df.color, fill=False)
    #ax.barh(df.task, df.calc_time, left=df.latest_start, color=df.color, fill=False, hatch='///')

    rownum = 0
    current_batch = ''
    for index, row in df.iterrows():
        #if(row.task_status == 'Leader'):
        #    ax.text(row.start-0.1, rownum, row.batch, va='center', alpha=0.8, ha='right', fontsize=8)
        if (row.batch != current_batch):  
            ax.text(row.start-0.1, rownum, row.batch, va='center', alpha=0.8, ha='right', fontsize=10)
            
            #ax.text(row.start-0.1, rownum, row.task, va='center', ha='right', alpha=0.7)
            current_batch = row.batch
        rownum += 1


    # grid lines
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.2, which='both')

    # ticks
    xticks = np.arange(0, df.end.max()+1, 50)
    xticks_labels = pd.date_range(0, end=df.end.max()).strftime("%m/%d")
    xticks_minor = np.arange(0, df.end.max()+1, 10)
    ax.set_xticks(xticks)
    ax.set_xticks(xticks_minor, minor=True)
    ax.set_yticks([])

    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['top'].set_visible(False)

    plt.suptitle('SCHEDULE')

    ##### LEGENDS #####
    legend_elements = [Patch(facecolor='#E64646', label='Leader'),
                    Patch(facecolor='#34D05C', label='Batch')]

    ax1.legend(handles=legend_elements, loc='upper center', ncol=5, frameon=False)

    # clean second axis
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.show()


runExperiment()