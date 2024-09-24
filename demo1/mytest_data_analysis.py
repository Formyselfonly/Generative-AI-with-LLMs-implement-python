import pandas as pd
from datasets import load_dataset

dataset_name="knkarthick/dialogsum"
dataset=load_dataset(dataset_name)
print("dataset:\n",dataset)

df_test=pd.DataFrame(dataset["test"])
print(df_test.index,df_test.columns,df_test.head(),df_test.describe(),sep="--------------------------------\n")

data_indecies=[0,100]
dashline="-"*20
for i,index in enumerate(range(data_indecies[0],data_indecies[-1])):
    print(index+1)
    print("Dialogue: ",dataset["test"][index]["dialogue"])
    print(dashline)
    print("Summary: ",dataset["test"][index]["summary"])
