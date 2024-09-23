from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import GenerationConfig
import pandas as pd
# pd.set_option('display.max_columns', None)

dashline="-"*100
# Load dataset
dataset_name="knkarthick/dialogsum"
dataset=load_dataset(dataset_name)
print("dataset\n",dataset)

df_train=pd.DataFrame(dataset["train"])
df_validation=pd.DataFrame(dataset["validation"])
df_test=pd.DataFrame(dataset["test"])

print(dashline,"\ndf_train\n")
print(df_train.head())
print(df_train.columns)
print(df_train.describe())

print(dashline,"\ndf_validation\n")
print(df_validation.head())
print(df_validation.columns)
print(df_validation.describe())

print(dashline,"\ndf_test\n")
print(df_test.head())
print(df_test.columns)
print(df_test.describe())
# print(dataset[1]["diague"])
