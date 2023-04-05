import pandas as pd

df = pd.read_csv("./still.csv", index_col=0, usecols=range(4))
end = df.index[-1]
df = df.loc[(df.index >= 30) & (df.index <= end - 30)]

count = 0
for start in range(30, int(end) - 30, 12):
    d = df.loc[(df.index >= start) & (df.index <= start + 12)]
    if len(d) == 0 or (d.index[-1] - d.index[0] <= 10):
        continue
    count += 1

    min = d.index[0]
    d.index = d.index.map(lambda x: x - min)
    d.to_csv(f'.././dataset_still/still_{count}.csv')

print(count)
