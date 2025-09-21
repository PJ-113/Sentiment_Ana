import pandas as pd
import matplotlib.pyplot as plt

# โหลดไฟล์ csv ที่ได้จากการ train
df = pd.read_csv("metrics_per_epoch.csv")

# วาดตารางเป็นรูป
fig, ax = plt.subplots(figsize=(8, 2))
ax.axis("off")
tbl = ax.table(cellText=df.values,
               colLabels=df.columns,
               cellLoc="center",
               loc="center")

tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)

plt.savefig("metrics_table.png", dpi=150, bbox_inches="tight")
plt.show()
