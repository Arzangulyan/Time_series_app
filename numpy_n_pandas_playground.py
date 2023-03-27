import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])

x.ndim
x.shape
x.size
x.dtype
x.itemsize

y = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], dtype="uint8")
y = np.array([1,2,3], dtype="uint8")
y[0] = 1001
print(y)

z = np.zeros((4,5))
print (z)
print()
z = np.zeros((4,5), dtype="int64")
print (z)

a = np.arange(1,100, np.pi)

np.linspace(1,100, 15)

y = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15], [16,17,18]], dtype="uint8")
y
y.shape
y.reshape(2,6)

y.resize(3,2,3)
y

a = np.array([[1,2,3],[4,5,6]])
b = np.array([[1,1],[1,1],[1,1]])
a
b
a@b

print(b.transpose()/a)

a = np.arange(1,21).reshape(4,5)
print(a)
print (a.transpose())

print(np.rot90(a,-4))

print(a.sum(axis=1).shape)
print(a.max())
print(a.min())

a = np.arange(1,21).reshape(4,5)
print(a)
print(a[:2,2:])
print(a[1:3,1:3])

a = np.arange(1,21).reshape(4,5)
a = np.arange(1, 13).reshape(3, 4)
print(a)
for row in a:
    print(row)
    for element in row:
        print(element)

print("; ".join(str(el) for el in a.flat))

from time import time

t = time()
print(f"Результат итератора: {sum(x ** 0.5 for x in range(10 ** 7))}.")
t1 = time() - t
print(f"{time() - t} с.")
t = time()
print(f"Результат numpy: {np.sqrt(np.arange(10 ** 7)).sum()}.")
t2 = time() - t
print(f"{time() - t} с.")

print (t1/t2)

print(np.linspace(0,1,5))

# Начинаем работать с Pandas

d = {"a": 10, "b": 20, "c": 30, "g": 40}
print(pd.Series(d))
print()
print(pd.Series(d, index=["a", "y", "g"]))

index = ["a", "b", "c", 't']
print(pd.Series(6, index=index))

s = pd.Series(np.arange(7), index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])
print (s)
print(s[s%3==2])


dict_sex = {
    'Country': ['Russia', 'Armenia', 'USA', 'UK'],
    'Man': [200, 100, 50, 25],
    'Woman': [1000, 500, 250, 125]}
print(dict_sex)
df = pd.DataFrame(dict_sex)
print (df)

print(df.index)
print(df.columns)

print(df.loc[2:])

print(type(df))
print(type(df['Woman']))

print(df[0:-2])

df = pd.read_csv('/Users/arzangulyan/Downloads/Students_Performance_132b1e1ff9.csv')
print(df)

print(df.head())

print(df[df["test preparation course"] == "completed"]["writing score"])

completers = df[df["test preparation course"] == "completed"]

completers[["math score", "reading score", "writing score"]].sort_values(by=["math score", "reading score", "writing score"], ascending=False).head()


with_course = df[df["test preparation course"] == "completed"]
df["total score"] = df["math score"] + df["reading score"] + df["writing score"]
print(df.sort_values(["total score"], ascending=False))

scores = df.assign(tot_score = lambda x: x["math score"] + x["reading score"] + x["writing score"])
print(scores.sort_values("tot_score",ascending=False))

df.groupby(["gender", "test preparation course"])["math score"].count()

agg_func = {"total score": ["mean", "median"]}
df.groupby(["gender", "test preparation course"]).agg(agg_func)

plt.hist(df["math score"], label="Math test", bins=100)
plt.show()
help(plt.hist)

df = pd.read_csv("/Users/arzangulyan/Documents/Научка/Vesna2022/Programming/Attempt_1/Vietnam_CO2_Temp.csv")

original_time = df["Time"]

pd.to_datetime(original_time, format="%m/%d/%y %H:%M")

help(print)
