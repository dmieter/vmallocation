import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pandas import Timestamp

##### DATA PREP ##### 
df = pd.read_csv('tasks.csv', delimiter=';')
print(df)

df["status"] = 'None'

# days between start and end of each task
df['interval'] = df.end - df.start

# days between start and current progression of each task
df['calc_days'] = (df.volume / 1)
df['latest_start'] = (df.end - df.calc_days)
df['earliest_finish'] = (df.start + df.calc_days)



def assignToLeader(df, time, num):
    df.loc[df.status == 'None', "possible_start"] = np.maximum(df.start, time)
    tdf = df[df.status == 'None']  # next available tasks to schedule

    tdf.earliest_finish = tdf.possible_start + df.calc_days
    EFT = tdf.earliest_finish.min()
    
    df.loc[(tdf.earliest_finish == EFT) & (df.status == 'None'), ['status', 'batch', 'finish_time']] = 'Leader', 'Batch ' + str(num), EFT
    df.loc[(tdf.latest_start < EFT) & (df.status == 'None'), ['status', 'batch', 'finish_time']] = 'Batch', 'Batch ' + str(num), EFT
    print(EFT)

    return EFT

def retrieveParalellBatches(df, time):
    batch_num = 0
    while('None' in df.status.values):
        time = assignToLeader(df, time, batch_num)
        batch_num += 1

retrieveParalellBatches(df, 0)

df = df.sort_values(['finish_time', 'status', 'start'], ascending = [True, False, True])
print(df)

# create a column with the color for each department
def color(row):
    c_dict = {'Leader':'#E64646', 'CPU 2':'#E69646', 'Batch':'#34D05C', 'CPU 4':'#34D0C3', 'CPU 5':'#3475D0', 'None' : '#000000'}
    return c_dict[row['status']]

df['color'] = df.apply(color, axis=1)

##### PLOT #####
fig, (ax, ax1) = plt.subplots(2, figsize=(16,6), gridspec_kw={'height_ratios':[6, 1]})

# bars

#ax.barh(df.Task, df.calc_days, left=df.LST, color=df.color, fill=False, hatch='---')
ax.barh(df.task, df.interval, left=df.start, color=df.color, alpha=0.3)
ax.barh(df.task, df.calc_days, left=df.possible_start, color=df.color)
#ax.barh(df.Task, df.calc_days, left=df.start_num, color=df.color, fill=False)
ax.barh(df.task, df.calc_days, left=df.latest_start, color=df.color, fill=False, hatch='///')

rownum = 0;
for index, row in df.iterrows():
    ax.text(row.end+0.1, rownum, row.batch, va='center', alpha=0.8)
    ax.text(row.start-0.1, rownum, row.task, va='center', ha='right', alpha=0.7)
    rownum += 1


# grid lines
ax.set_axisbelow(True)
ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.2, which='both')

# ticks
xticks = np.arange(0, df.end.max()+1, 5)
xticks_labels = pd.date_range(0, end=df.end.max()).strftime("%m/%d")
xticks_minor = np.arange(0, df.end.max()+1, 1)
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