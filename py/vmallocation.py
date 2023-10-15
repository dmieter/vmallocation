# %% IMPORTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch    

print("Import Successful")

# %% LOAD INPUT DATA

##### LOAD TASKS ##### 
def loadTasks():
    tasks = pd.read_csv('tasks.csv', delimiter=';')
    data_transfers = pd.read_csv('datatransfers.csv', delimiter=';')

    tasks['index'] = tasks['task']
    tasks.set_index('index', inplace = True)

    tasks["task_type"] = 'Task'
    tasks["task_status"] = 'None'

    # days between start and end of each task
    tasks['interval'] = tasks.end - tasks.start

    return tasks, data_transfers

##### LOAD VMs ##### 
def loadVMs():
    vm_counter = 0
    vm_types = pd.read_csv('vmtypes.csv', delimiter=';')
    vms = pd.DataFrame()
    vms_log = pd.DataFrame()
    print("VM TYPES:")
    print(vm_types)

    return vm_types, vms, vms_log


tasks, data_transfers = loadTasks()
vm_types, vms, vms_log = loadVMs()

print("""
      
TASKS:""")
print(tasks)
print("""

DATA TRANSFERS:""")
print(data_transfers.head())


# %% SEPARATE TASK IN BATCHES
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
    
    tasks.loc[(ttasks.earliest_finish == EFT) & (tasks.task_status == 'None'), ['task_status', 'batch', 'finish_time']] = 'Leader', 'Batch ' + str(num), EFT
    tasks.loc[(ttasks.latest_start < EFT) & (tasks.task_status == 'None'), ['task_status', 'batch', 'finish_time']] = 'Batch', 'Batch ' + str(num), EFT
    #print(EFT)

    return EFT

def retrieveParalellBatches(vm_types, tasks, time):
    batch_num = 0
    while('None' in tasks.task_status.values):
        time = assignToLeader(vm_types, tasks, time, batch_num)
        batch_num += 1


#retrieveParalellBatches(vm_types, tasks, 0)

#tasks = tasks.sort_values(['finish_time', 'task_status', 'start'], ascending = [True, False, True])
#print(tasks)        


# %% HELPER METHODS

def addOffTasks(batch, num, counter = 0):
    off_tasks = {'task': ['Off ' + str(i+counter) for i in range(num)],
                'task_type' : ['Off' for i in range(num)]}
    
    off_tasks_df = pd.DataFrame(off_tasks)
    
    return pd.concat([batch,off_tasks_df], axis=0, ignore_index=True)

def generateVmId(row):
    global vm_counter
    vm_counter = vm_counter + 1
    return 'VM ' + str(vm_counter)

def addNewVmsRandomly(vms, vm_types, num):
    random_types = vm_types.sample(num, replace = True)
    random_types['vm_status'] = 'open'
    random_types['vm_id'] = random_types.apply(lambda x: generateVmId(x), axis=1)  
    
    return pd.concat([vms, random_types], axis=0, ignore_index=True)

#batch = addOffTasks(batch, 10)

#vms = addNewVmsRandomly(vms, vm_types, 5)

def clearTasks(batch):
    return batch[batch.task_type != 'Off']

def getActiveVms(vms):
    return vms[vms.vm_status == 'active']

def prepareVmMatchings(batch, vms, additional_vms_num = 0):
    # first remove old temp tasks and not started vms
    if not batch.empty:
        batch = clearTasks(batch)
    if not vms.empty:
        vms = getActiveVms(vms)
    
    diff = len(batch) - len(vms)
    if diff > 0:
        vms = addNewVmsRandomly(vms, vm_types, diff)
    elif diff < 0:
        batch = addOffTasks(batch, -diff)
        
    # add additional vms and tasks for more optimization options    
    if additional_vms_num > 0:
        vms = addNewVmsRandomly(vms, vm_types, additional_vms_num)
        batch = addOffTasks(batch, additional_vms_num, len(batch) + 1)  # use len(batch) + 1 to avois name collisions

    #pair all    
    vms['key'] = 1
    batch['key'] = 1
    matching = pd.merge(batch, vms, on = 'key', suffixes=("_task", "_vm"))
    
    #print(matching)
    
    return matching
        

#matchings = prepareVmMatchings(batch, vms, 2)
#print(matchings)

# %% PAIRING ALGORITHMS

import random
import math

current_time = 0
DATA_TRANSFER_SPEED = 50

def calcVmAllocationCost(row):

    #init
    
    idle_time = 0                   # time vm idle between end of previous task and start of current task (len)
    preparation_time = 0            # time needed to prepare vm to start task (usually start vm + get data) (len)
    runtime = 0                     # actual task runtime (len)
    release_time = 0                # time needed to cleanup vm and copy its data before turn off (len)
    max_data_transfer_time = 0      # max time needed to transfer all data from all source tasks (len)

    earliest_data_ready_time = 0    # earliest time all data can be copied from source tasks (moment)

    

    vm_type = vm_types[vm_types.vm_type == row.vm_type].iloc[0]

    # if new vm
    if(row.vm_status == 'open'):
        possible_vm_start = current_time  # can start now
    else:
        possible_vm_start = tasks[tasks.assigned_vm == row.vm_id].allocation_end.max()  # can start once previous tasks are finished
        

    
    # if turning off
    if row.task_type == 'Off':
        possible_task_start = current_time                      # release task can be started any time (no data/logic restrictions)

        release_time = vm_type.vm_stop_time 
        outgoing_data = data_transfers[data_transfers.source_task == row.task]
        if len(outgoing_data) > 0:
            outgoing_data['transfer_time'] = outgoing_data.volume/DATA_TRANSFER_SPEED
            outgoing_data_transfer_time = outgoing_data.transfer_time.max()
            release_time = release_time + outgoing_data_transfer_time
        
        
    
    # if perform calculations
    else:
        runtime = math.ceil(row.volume / vm_type.perf) 
        if(row.vm_status == 'open'):
            preparation_time = preparation_time + vm_type.vm_start_time             # add startup time

        required_data = data_transfers[data_transfers.target_task == row.task]
        required_data['transfer_time'] = required_data.volume/DATA_TRANSFER_SPEED  # maybe speed should be decreased based on number of parallel data transfers
        if len(required_data) > 0:
            required_data = pd.merge(required_data, tasks, left_on = 'source_task', right_on = 'task')[['task', 'transfer_time', 'allocation_end', 'assigned_vm']]
            required_data[required_data.assigned_vm == row.vm_id]['transfer_time'] = 0
            required_data['earliest_ready_time'] = required_data.allocation_end + required_data.transfer_time
            earliest_data_ready_time = required_data.earliest_ready_time.max()
            max_data_transfer_time = required_data.transfer_time.max()
            preparation_time = preparation_time + max_data_transfer_time       

        possible_task_start = max(row.start, earliest_data_ready_time)    # possibly check if can start earlier, i.e. remove row.start from max



    full_runtime = preparation_time + runtime + release_time

    expected_vm_start = max(possible_vm_start, possible_task_start - preparation_time)
    expected_vm_end = expected_vm_start + full_runtime
    if(row.vm_status != 'open'):
        idle_time = (expected_vm_start - possible_vm_start)     # idle time between previous assignment and new assignment

    expected_task_start = expected_vm_start + preparation_time
    expected_task_end = expected_task_start + runtime
    
    if expected_task_end > row.end:
        allocation_cost = 10000000000 # can't execute task
    else:    
        allocation_cost = (full_runtime + idle_time) * vm_type.cost
        
    return allocation_cost + 1, expected_vm_start, expected_task_start, expected_task_end, expected_vm_end  # allocation_cost + 1 because zeros are bad for optimization

def calcAllocationCosts(matchings):
    matchings[['allocation_cost', 'vm_start', 'task_start', 'task_end', 'vm_end']] = matchings.apply(lambda x: calcVmAllocationCost(x), axis=1, result_type='expand')  
    

def calcPairing(row, task_list, vm_list):
    if row.task in task_list and row.vm_id in vm_list:
        task_list.remove(row.task)
        vm_list.remove(row.vm_id)
        return 1
    else:
        return 0


from hungarian_algorithm import algorithm

def calculateMinCostPairings(matchings):
    unique_tasks = list(matchings.task.unique())
    cost_matrix = {}
    for task_name in unique_tasks:
        task_pairs = matchings[matchings.task == task_name][['vm_id', 'allocation_cost']]
        task_pairs.set_index('vm_id')
        pair_data = task_pairs.to_dict('split')['data']
        task_dict = {}
        for pair in pair_data:
            task_dict[pair[0]] = pair[1]
        cost_matrix[task_name] = task_dict

    print(cost_matrix)
    res = algorithm.find_matching(cost_matrix, matching_type = 'max', return_type = 'list' )
    matchings['pairing'] = 0
    for task_res in res:
        matchings.loc[(matchings.task == task_res[0][0]) & (matchings.vm_id == task_res[0][1]), 'pairing'] = 1
    
    pairings = matchings[matchings.pairing == 1]
    
    return pairings

from munkres import Munkres, print_matrix, DISALLOWED
def calculateMinCostPairings2(matchings):
    unique_tasks = list(matchings.task.unique())
    unique_vms = list(matchings.vm_id.unique())
    cost_matrix = []
    for task_name in unique_tasks:
        task_vms_list = [DISALLOWED] * len(unique_vms)

        task_pairs = matchings[matchings.task == task_name][['vm_id', 'allocation_cost']]
        task_pairs.set_index('vm_id')
        pair_data = task_pairs.to_dict('split')['data']
        
        for pair in pair_data:
            vm_index = unique_vms.index(pair[0])
            if(pair[1] < 10000000000):
                task_vms_list[vm_index] = pair[1]
            else:    
                task_vms_list[vm_index] = DISALLOWED

        cost_matrix.append(task_vms_list)

    print_matrix(cost_matrix)  

    m = Munkres()
    result = m.compute(cost_matrix)
    print(result)
    for row, column in result:
        task_name = unique_tasks[row]
        vm_id = unique_vms[column]
        matchings.loc[(matchings.task == task_name) & (matchings.vm_id == vm_id), 'pairing'] = 1

    pairings = matchings[matchings.pairing == 1]
    return pairings



# %% SCHEDULE AND ALLOCATION

def applyPairings(pairings, tasks, vms, vms_log):
    
    vms_off = pairings[(pairings.vm_status == 'active') & (pairings.task_type == 'Off')]
    vms_off = vms_off[['vm_id', 'vm_status', 'vm_end']]
    vms_off['vm_status'] = 'shutdown'
    vms_off['index'] = vms_off['vm_id']
    vms_off.set_index('index', inplace = True)
    vms = vms_off.combine_first(vms)
    
    vms_cancel = pairings[(pairings.vm_status == 'open') & (pairings.task_type == 'Off')]  # do nothing, these vms weren't added to vms df
    
    vms_activate = pairings[(pairings.vm_status == 'open') & (pairings.task_type == 'Task')]
    vms_activate = vms_activate[['vm_id', 'vm_status', 'vm_type', 'perf', 'vm_start']]
    vms_activate['vm_status'] = 'active'
    vms_activate['index'] = vms_activate['vm_id']
    vms_activate.set_index('index', inplace = True)
    vms = vms_activate.combine_first(vms)
   
    vms_reuse = pairings[(pairings.vm_status == 'active') & (pairings.task_type == 'Task')] # do nothing, they just remain active

    pairing_log = pairings[(pairings.vm_status != 'open') | (pairings.task_type != 'Off')][['vm_id', 'task', 'vm_start', 'task_start', 'task_end', 'vm_end', 'allocation_cost']]
    pairing_log.allocation_cost = pairing_log.allocation_cost - 1 # removing extra 1 required for hungarian alg
    vms_log = vms_log.append(pairing_log)
    
    assigned_tasks = pairings[pairings.task_type == 'Task']
    assigned_tasks = assigned_tasks[['task', 'task_start', 'task_end', 'vm_id']]
    assigned_tasks.rename(columns={'vm_id':'assigned_vm', 'task_start':'allocation_start', 'task_end':'allocation_end'}, inplace=True)
    assigned_tasks['index'] = assigned_tasks['task']
    assigned_tasks.set_index('index', inplace = True)
    tasks = assigned_tasks.combine_first(tasks)
    
    print("VMS:")
    print(vms)
    
    #print("TASKS:")
    #print(tasks)

    return tasks, vms, vms_log
    
#calcAllocationCosts(matchings)
#pairings = simulatePairings(matchings)
#print(pairings[["task", 'volume', "vm_id", "vm_status", "allocation_cost", 'allocation_start', 'allocation_end', "pairing"]])
#applyPairings(pairings)
#print(pairings)


def scheduleBatch(batch_num, tasks, vms, vms_log):
    print("Time: {} Batch: {}".format(current_time, batch_num))
    batch = tasks[tasks.batch == batch_num]
    matchings = prepareVmMatchings(batch, vms, 2)
    calcAllocationCosts(matchings)
    pairings = calculateMinCostPairings2(matchings)
    print(pairings[["task", 'volume', "vm_id", "allocation_cost", 'vm_start', 'task_start', 'task_end', 'vm_end', "pairing"]])
    tasks, vms, vms_log = applyPairings(pairings, tasks, vms, vms_log)

    return tasks, vms, vms_log


def runExperiment():
    global tasks, data_transfers, vm_types, vms, vm_counter
    tasks, data_transfers = loadTasks()
    vm_types, vms, vms_log = loadVMs()
    vm_counter = 0

    retrieveParalellBatches(vm_types, tasks, 0)
    tasks = tasks.sort_values(['finish_time', 'task_status', 'start'], ascending = [True, False, True])

    batches = list(tasks.batch.unique())

    for batch_num in batches:
        tasks, vms, vms_log = scheduleBatch(batch_num, tasks, vms, vms_log)

    print("""
          
    VMS:""")
    print(vms)
        
    print("""
          
    TASKS:""")
    print(tasks)

    print("""
          
    ALLOCATION LOG:""")
    print(vms_log)


# %% TEST

runExperiment()