from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import GenerationConfig

# Load dataset
dataset_name="knkarthick/dialogsum"
dataset=load_dataset(dataset_name)
print(dataset)

# Test
example_indices=[40,200]
dashline="-"*100
# dashline100_1="-"*100
# print("dashline100_1",dashline100_1)
# dashline100_2="-".join('' for i in range(101))
# print("dashline100_2",dashline100_2)
# print(dashline100_1==dashline100_2)

for i,index in enumerate(example_indices):
    print(dashline)
    print("Example",i+1)
    print(dashline)
    print("INPUT DIALOGUE\n",dataset["test"][index]['diague'])
    print(dashline)
    print("Output Summary:",dataset["test"][index]["summary"])
    print(dashline)
    print()