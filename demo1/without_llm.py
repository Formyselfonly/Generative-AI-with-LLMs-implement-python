from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import GenerationConfig

# Load dataset
dataset_name="knkarthick/dialogsum"
dataset=load_dataset(dataset_name)
# print(dataset)

# Test
example_indices=[40,200]
dashline="-"*100
# dashline100_1="-"*100
# print("dashline100_1",dashline100_1)
# dashline100_2="-".join('' for i in range(101))
# print("dashline100_2",dashline100_2)
# print(dashline100_1==dashline100_2)

for i,index in enumerate(range(example_indices[0],example_indices[-1])):
    print(dashline)
    print("Example",i+1)
    print("id:",dataset["test"][index]["id"])
    print("Topic:",dataset["test"][index]["topic"])
    print("INPUT DIALOGUE\n",dataset["test"][index]['dialogue'])
    print(dashline)
    print("Output Summary:",dataset["test"][index]["summary"])
    print(dashline)
    print()

model_name="google/flan-t5-base"
model=AutoModelForSeq2SeqLM.from_pretrained(model_name,cache_dir="./cache/model")
tokenizer=AutoTokenizer.from_pretrained(model_name,cache_dir="./cache/tokenizer",use_fast=True)

# sentence="How to play badminton's clear and lift?"
# sentence_encoded=tokenizer(sentence,return_tensors="pt")
# print("sentence_encoded",sentence_encoded)
# sentence_decoded=tokenizer.decode(
#     sentence_encoded["input_ids"][0],
#     skip_special_tokens=True,
#     clean_up_tokenization_spaces=True
# )
# print('ENCODED SENTENCE:')
# print(sentence_encoded["input_ids"][0])
# print('\nDECODED SENTENCE:')
# print(sentence_decoded)


for i,index in enumerate(range(example_indices[0],example_indices[-1])):
    dialogue=dataset["train"][index]["dialogue"]
    summary=dataset["train"][index]["summary"]
    inputs=tokenizer(dialogue,return_tensors="pt")
    output=tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=100,
        )[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    print(dashline)
    print('Example ', i + 1)
    print(dashline)
    print(f'INPUT PROMPT:\n{dialogue}')
    print(dashline)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dashline)
    print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING:\n{output}\n')