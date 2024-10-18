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
import json

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


character_prompt = """
您的任务是，首先是分类以下文档，回答属于以下类别哪一种：财务/实体描述/医疗/政策/科学/论文摘要/百科知识/电影/百度知识/气候/疫情/网络/电子商务/医学/其他。其次是根据以下文档内容创造一个角色，可从以下选择但不限于：病人/医生/金融工作者/学生/公务员/科学家/研究者/普通人/买家/卖家/演员/观众，这个角色最有可能发现以下文档的价值。最后为这个角色设想一个最可能利用以下文档的常见的场景。
请遵守以下准则：
- 请注意，仅回答类别、角色和场景，生成的场景不超过10个字，不要回答其他内容。
- 输出格式，类别：xx。角色：xx。场景：xx。其中，xx为占位符，具体内容待生成。
文档：\n\n{},
您的输出必须始终是一个字符串，发挥创意！"""


query_prompt = """
您正在构建文本检索任务的示例，一个文本检索示例必须包含以下两个部分：
- "user_query"：\n\n一个10字左右的字符串，与"positive_document"的内容密切相关。
- "positive_document"：\n\n一个文本段落，"user_query"必须可以在"positive_document"找到答案，"positive_document"应该是"user_query"最相关的文档。
其中，"positive_document"：\n\n{},
您的身份为{}，所处的场景为{}，您已经阅读了"positive_document"的内容，根据以上要求生成一个{}“user_query”, 请注意"user_query"仅为一个单独的{}，不要包含多个{}"。
您的输出必须始终是一个字符串，不要解释自己或输出任何其他内容。"""

query_prompt_02 = """
您正在构建文本检索任务的示例，一个文本检索示例必须包含以下两个部分：
- "user_query"：\n\n一个10字左右的字符串，与"positive_document"的内容密切相关。
- "positive_document"：\n\n一个文本段落，"user_query"必须可以在"positive_document"找到答案，"positive_document"应该是"user_query"最相关的文档。
其中，"positive_document"：\n\n{},
您已经阅读了"positive_document"的内容，根据以上要求生成一个单独的通用搜索查询问题“user_query”, 请注意"user_query"仅为一个单独的问题，不要包含多个问题"。
您的输出必须始终是一个字符串，不要解释自己或输出任何其他内容。"""

query_prompt = system_prompt + "文档2:{}"

model_path = "/path/models--Qwen--Qwen2-72B-Instruct/snapshots/1af63c698f59c4235668ec9c1395468cb7cd7e79"
file_path = sys.argv[1]
local_rank = int(sys.argv[2])
batch_size = 2048
portioning = 2 # 一个json文件，可用多少张卡/4就可分成几份。

# type1 加载文件夹，type2 加载json文件
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
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=2000)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model=model_path, tensor_parallel_size=4, max_model_len=2000, enable_prefix_caching=True)

if os.path.isdir(file_path):
    data_path_base = file_path_r_list[0]+"_merge"
else:
    data_path_base = file_path
    

with open(data_path_base+f"_{local_rank}_qwen2_72b.json", 'w') as jsonl_file_w:
    total_doc_sub = total_doc_res[local_rank::portioning]
    for start_index in tqdm(range(0, len(total_doc_sub),batch_size), desc="Batches", disable=len(total_doc_sub)<256):
        data_batch = total_doc_sub[start_index:start_index + batch_size]
        data_batch_character_prompt = [character_prompt.format(data["pos"][0][:512]) for data in data_batch]
        
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
        # generate outputs
        character_outputs = llm.generate(character_token_batch, sampling_params)
        chars = [i.outputs[0].text for i in character_outputs]
        try:
            chars_dict = [{item.split('。')[0].split("：")[0]:item.split('。')[0].split("：")[1], item.split('。')[1].split("：")[0]:item.split('。')[1].split("：")[1],item.split('。')[2].split("：")[0]:item.split('。')[2].split("：")[1]} for item in chars]
            data_batch_query_prompt = [query_prompt.format(data, char["角色"], char["场景"], instr.get(char["类别"],"问题"),instr.get(char["类别"],"问题"),instr.get(char["类别"],"问题")) for data,char in zip(data_batch,chars_dict)]
            
        except:
            data_batch_query_prompt = [query_prompt_02.format(data) for data in data_batch]

        token_batch = []
        for data in data_batch_query_prompt:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": data}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            token_batch.append(text)
        # generate outputs
        outputs = llm.generate(token_batch, sampling_params)
        
        for index,output in enumerate(outputs):
            try:
                generated_text = output.outputs[0].text
                generated_text = generated_text.split("：")[-1]
                if len(generated_text) > 0 and len(data_batch[index])>0:
                    data_ = {"query":"","pos":[],"neg":[]}
                    data_["query"] = generated_text
                    data_["pos"] = [data_batch[index]]
                    jsonl_file_w.write(json.dumps(data_) + '\n')
            except:
                pass


