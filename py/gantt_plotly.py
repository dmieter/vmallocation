import pandas as pd
import plotly.express as px

dl_df = pd.DataFrame([
    dict(Model="EfficientNetB4", Start='2021-08-05 08:00:00', Finish='2021-08-05 15:24:21'),
    dict(Model="VGG16", Start='2021-08-05 15:25:00', Finish='2021-08-05 17:30:00'),
    dict(Model="DenseNet121", Start='2021-08-05 17:30:00', Finish='2021-08-05 21:45:00'),
    dict(Model="InceptionV3", Start='2021-08-05 21:45:00', Finish='2021-08-06 05:20:00'),
    dict(Model="MobileNet", Start='2021-08-06 05:20:00', Finish='2021-08-06 06:00:00')
])

df = pd.DataFrame([
    dict(Task="Task 1", Start='2021-01-01', Finish='2021-02-28', Resource="Process 1"),
    dict(Task="Task 2", Start='2021-03-01', Finish='2021-04-15', Resource="Process 2"),
    dict(Task="Task 3", Start='2021-04-15', Finish='2021-05-30', Resource="Process 1")
])

fig = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", color="Resource")
fig.show()