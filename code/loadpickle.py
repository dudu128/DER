import pickle
import pandas as pd

objects = []
with (open("./wrong_pickle/wrong_6.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

print (objects)
# len of data
print (len(objects[0]))
# len of feature per data
print (len(objects[0][0]))

# if label = 6 
data=[6]*len(objects[0])
df = pd.DataFrame(data)
df.columns = ["label"]

# print(df.shape)
print(df)
df.to_csv('out.csv')
