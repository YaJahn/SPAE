import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from collections import defaultdict
# % matplotlib inline
import matplotlib
matplotlib.rcParams.update({'font.size': 25})  

df = pd.read_csv(r'F:\my_results\1_mesc_Quartz\my_evaluation_mesc_Quartz_avg.csv') # mesc_Quart
# df = pd.read_csv(r'F:\my_results\2_mesc_288\evaluation_mesc_avg_my.csv') # mesc_288


groups = df.groupby("method")


chart = plt.figure(figsize=(15, 15)).add_subplot(polar=True)
# chart.set_facecolor('none')

labels = list(df["metrics"].unique())
angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
angles += angles[:1]
chart.set_theta_offset(pi / 2)
chart.set_theta_direction(-1)

plt.xticks(angles[:-1], labels)
# plt.xticks(angles[:-1], [])  


colors =["blue","green",'#FFA500',"#800080","red","bisque"]
c = 0

for name, group in groups:
    values = defaultdict(list)
    print("name:",name)
    # print("group:",group)
    for i, row in group.iterrows():
        values[row["metrics"]].append(row["value"])
    values = [sum(values[label]) / len(values[label]) for label in labels]

    values += values[:1]

    chart.plot(angles, values, linewidth=2, color=colors[c],label=name)
    chart.fill(angles, values, alpha=0.25)
    c = c+1


chart.legend(loc="upper right",framealpha=0.8, bbox_to_anchor=(1.2, 1.0))
# ax.legend(framealpha=0.8, facecolor='lightgray')
# plt.savefig('mesc_Quartz.tiff', dpi=600, format='tiff')

plt.show()

