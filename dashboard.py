import os
import pandas as pd
import matplotlib.pyplot as plt

LOG_FILE = "alerts_log.csv"

if not os.path.exists(LOG_FILE):
    print("alerts_log.csv not found. Run step8.py and trigger some alerts first.")
    raise SystemExit

df = pd.read_csv(LOG_FILE)
if df.empty:
    print("The log is empty. Trigger some alerts first.")
    raise SystemExit

df["time"] = pd.to_datetime(df["time"])

colors = {
    "DROWSY!": "#e74c3c",
    "YAWNING!": "#f39c12",
    "LOOK AT THE ROAD!": "#3498db",
    "NO FACE FOUND": "#7f8c8d",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("AI Driver Alert System - Dashboard", fontsize=15, fontweight="bold")

# Chart 1: alerts by type
counts = df["alert"].value_counts()
ax1.bar(counts.index, counts.values,
        color=[colors.get(a, "#95a5a6") for a in counts.index])
ax1.set_title(f"Alerts by type (total: {len(df)})")
ax1.set_ylabel("Number of alerts")
ax1.tick_params(axis="x", rotation=15)
for i, v in enumerate(counts.values):
    ax1.text(i, v + 0.05, str(v), ha="center", fontweight="bold")

# Chart 2: alerts over time
for alert, group in df.groupby("alert"):
    ax2.scatter(group["time"], [alert] * len(group), s=90,
                color=colors.get(alert, "#95a5a6"))
ax2.set_title("Alerts over time")
ax2.tick_params(axis="x", rotation=30)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("dashboard.png", dpi=150)   # saved image for your README / LinkedIn
print("Saved dashboard.png")
plt.show()