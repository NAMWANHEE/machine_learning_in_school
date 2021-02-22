import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
nba = pd.read_csv("nba.csv")
c = nba.pivot_table(["Age","Salary"],index=["Position","Team"])
print(c)
plt.figure(figsize=(5,5))
b = nba.groupby("Position")
max_Salary = lambda g: g.Salary.max()
#print(max_Salary(b))
#nba.groupby("Position").apply(max_Salary)
h = np.arange(len(b))
plt.bar(h,max_Salary(b).values)
plt.xticks(h,sorted(nba.Position.drop_duplicates().dropna().values))

y1 = nba.Team.drop_duplicates().dropna()
y2 = y1.str.split().str[1]
team_name=y1.str[0]+y2.str[0]
#print(team_name)

team_max_age=nba.groupby("Team").Age.max()
team_max_Salary=nba.groupby("Team").Salary.max()
#max_Salary(nba.groupby("Team")

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(2,1,1)
ax1.bar(sorted(team_name.tolist()),team_max_Salary)
plt.title("Maximum salary for each team")

ax2 = fig.add_subplot(2,1,2)
ax2.bar(sorted(team_name.tolist()),team_max_age)
plt.title("Maximum age for each team")
plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
plt.show()