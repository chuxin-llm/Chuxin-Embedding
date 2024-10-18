import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
import os
from qwen_generation_utils import make_context, decode_tokens
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import sys
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

character_prompt = """
请问这个query 是否可作为一个通用的搜索引擎查询吗？直接回答是或者不是。
query:\n\n{}。\n
"""

#请问这个query 是过于描述细节并且这个query 包含多个互相之间无关的问题或查询吗？直接回答是或者否。
#query:\n\n{}。\n
#"""

#请问这个query 是属于1.财务问题 2.与实体描述相关的查询 3.医疗问题 4.政策相关的问题 5.科学主张 6.科学论文标题 7.与百科知识相关的问题 8.影视 9.气候变化的主张 10.与COVID-19有关的查询 11.>通用搜索引擎查询 12.商品查询 13.医学问题 以上13种吗？直接回答是或者不是。\n

instr = {
    "财务": "财务问题",
    "实体描述": "与实体描述相关的查询",
    "医疗": "医疗问题",
    "其他": "问题",
    "政策": "政策相关的问题",
    "科学": "科学主张",
    "论文摘要": "科学论文标题",
    "百科知识": "与百科知识相关的问题",
    "电影": "电影标题",
    "气候": "气候变化的主张",
    "疫情": "与COVID-19有关的查询",
    "网络": "网络搜索查询",
    "电子商务": "商品查询",
    "医学": "医学问题",
}


model_path = "/path/models--Qwen--Qwen2-72B-Instruct-GPTQ-Int4/snapshots/6b82a333287651211b1cae443ff2d2a6802597b9"
file_path = sys.argv[1]
portioning = 8 # 一个json文件，可用多少张卡就可分成几份。
local_rank = int(sys.argv[2]) # 一个json文件，第几份
batch_size = 1024
seq_len = 300

total_doc_res = []
if os.path.isdir(file_path):
    file_path_r_list = os.listdir(file_path)
    file_path_r_list = [i for i in file_path_r_list if "json" in i]
    file_path_r_list = [os.path.join(file_path,i) for i in file_path_r_list]
    for i in file_path_r_list:
        with open(i, 'r') as jsonl_files:
            for line in tqdm(jsonl_files):
                data = json.loads(line)
                total_doc_res.append(data)
else:
    with open(file_path, 'r') as jsonl_files:
        for line in tqdm(jsonl_files):
            data = json.loads(line)
            total_doc_res.append(data)
            
            

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Pass the default decoding hyperparameters of Qwen2-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.01, top_p=1.0, repetition_penalty=1.05, max_tokens=seq_len)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model=model_path, tensor_parallel_size=1, max_model_len=seq_len, quantization="gptq", kv_cache_dtype="fp8_e5m2",enable_prefix_caching=True)

index_save = 0
if os.path.isdir(file_path):
    data_path_base = file_path_r_list[0]+"_merge"
else:
    data_path_base = file_path
    

with open(data_path_base+f"_{local_rank}_cleanquery.json", 'w') as jsonl_file_w:
    total_doc_sub = total_doc_res[local_rank::portioning]
    for start_index in tqdm(range(0, len(total_doc_sub),batch_size), desc="Batches", disable=len(total_doc_sub)<256):
        data_batch = total_doc_sub[start_index:start_index + batch_size]
        
        data_batch_character_prompt = [character_prompt.format(data["query"])[:seq_len] for data in data_batch]
        character_token_batch = []
        for data in data_batch_character_prompt:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": data}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            character_token_batch.append(text)
  
        character_outputs = llm.generate(character_token_batch, sampling_params)
         
        start_score = 0
    
        for index,data in enumerate(data_batch):
            if "是" == character_outputs[index].outputs[0].text[:1]:
                jsonl_file_w.write(json.dumps(data_batch[index]) + '\n')
            else:
                pass
                # data_batch[index]["query"] = character_outputs[index].outputs[0].text.split("\n")[0]
                # print(character_outputs[index].outputs[0].text)
                # print(data_batch[index]["query"])
                # breakpoint()
               
            

