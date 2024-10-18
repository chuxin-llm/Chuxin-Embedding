import os
import re
import json
import unicodedata
import sys
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


file_path = sys.argv[1] 

batch_size = 8000
rerank_score = "_threshold_pos1_neg0"
save_index = 0
no_save_index = 0


# dir or file
if os.path.isdir(file_path):
    file_path_r_list = os.listdir(file_path)
    file_path_r_list = [i for i in file_path_r_list if "json" in i]
    if not os.path.exists(os.path.join(file_path,"newclean")):
        os.mkdir(os.path.join(file_path,"newclean"))
    file_path_r_list = [os.path.join(file_path,i) for i in file_path_r_list]
    file_path_w_list = [os.path.dirname(i)+f"/newclean/rerank{rerank_score}_"+os.path.basename(i) for i in file_path_r_list]
else:
    if not os.path.exists(os.path.join(os.path.dirname(file_path),"newclean")):
        os.mkdir(os.path.join(os.path.dirname(file_path),"newclean"))
    file_path_w = os.path.dirname(file_path)+f"/newclean/rerank{rerank_score}_"+os.path.basename(file_path)
    file_path_w_list = [file_path_w]
    file_path_r_list = [file_path]


# load tokenzier and model
tokenizer = AutoTokenizer.from_pretrained('/path/models--BAAI--bge-reranker-v2-m3/snapshots/12e974610ba9083ed95f3edf08d7e899581f4de4')
model = AutoModelForSequenceClassification.from_pretrained('/path/models--BAAI--bge-reranker-v2-m3/snapshots/12e974610ba9083ed95f3edf08d7e899581f4de4').to("cuda").half()
model.eval()
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    model = torch.nn.DataParallel(model)

# rerank 
for file_path_r, file_path_w in zip(file_path_r_list, file_path_w_list):
    jsonl_file_r = open(file_path_r, 'r')
    total = [json.loads(line) for line in tqdm(jsonl_file_r)]
    # total = total[3024000:][local_rank::4]
    with open(file_path_w, 'w') as jsonl_file_w: 
        # 用rerank判断 , 阈值可根据数据情况设定
        # data["pos"] 中是否有假阳例（score<0), data["pos"]中只保留阳例（score>0）
        # data["neg"] 中是否有假阴例（score>0), data["neg"]中只保留阴例（score<0）
        print("开始bge-rerank 清洗假阳/阴例")
        for start_index in tqdm(range(0, len(total), batch_size), desc="Batches", disable=len(total)<32):
            data_batch = total[start_index:start_index + batch_size]
            new_data_batch = []
            pairs_pos = []
            pairs_neg = []
            for data in data_batch:
                data["pos"] = list(set(data["pos"]))
                data["neg"] = list(set(data["neg"]))
                data_pos_text = data["pos"]
                data_neg_text = data["neg"]
                for text in data_pos_text:
                    pairs_pos.append([data["query"],text])
                for text in data_neg_text:
                    pairs_neg.append([data["query"],text])
                new_data_batch.append(data)

            # pos score    
            scores_pos_list = []
            for start in tqdm(range(0, len(pairs_pos), batch_size), desc="subBatches", disable=len(pairs_pos)<32):
                data_batch_sub = pairs_pos[start:start + batch_size]
                with torch.no_grad():
                    inputs = tokenizer(data_batch_sub, padding=True, truncation=True, return_tensors='pt', max_length=512).to("cuda")
                    scores = model(**inputs, return_dict=True).logits.view(-1, )
                    scores_pos_list.extend(scores)
            if  len(scores_pos_list) == len(pairs_pos):
                pass
            else:
                break
            
            # neg score    
            scores_neg_list = []
            for start in tqdm(range(0, len(pairs_neg), batch_size), desc="subBatches", disable=len(pairs_neg)<32):
                data_batch_sub = pairs_neg[start:start + batch_size]
                with torch.no_grad():
                    inputs = tokenizer(data_batch_sub, padding=True, truncation=True, return_tensors='pt', max_length=512).to("cuda")
                    scores = model(**inputs, return_dict=True).logits.view(-1, )
                    scores_neg_list.extend(scores)
            if  len(scores_neg_list) == len(pairs_neg):
                pass
            else:
                break

            start_pos_score = 0
            start_neg_score = 0
            for index,data in enumerate(new_data_batch):
                # neg score
                data_neg_text = data["neg"]
                score_neg_tmp = scores_neg_list[start_neg_score:start_neg_score+len(data_neg_text)]
                start_neg_score = start_neg_score+len(data_neg_text)
                data["neg"] = [data_neg_text[i] for i, x in enumerate(score_neg_tmp) if x < 0]
                data["neg_scores"] = [x.cpu().item() for i, x in enumerate(score_neg_tmp) if x < 0]

                # pos score
                data_pos_text = data["pos"]
                score_pos_tmp = scores_pos_list[start_pos_score:start_pos_score+len(data_pos_text)]
                start_pos_score = start_pos_score+len(data_pos_text)
                data["pos"] = [data_pos_text[i] for i, x in enumerate(score_pos_tmp) if x > 1]
                data["pos_scores"] = [x.cpu().item() for i, x in enumerate(score_pos_tmp) if x > 1]
                neg = [data_pos_text[i] for i, x in enumerate(score_pos_tmp) if x < 0]
                neg_scores = [x.cpu().item() for i, x in enumerate(score_pos_tmp) if x < 0]
                # 假正例加入到负例中 
                data["neg"] = data["neg"][:15-len(neg)] + neg
                data["neg_scores"] = data["neg_scores"][:15-len(neg_scores)] + neg_scores
                if len(data["neg"])>0 and len(data["pos"])>0:
                    jsonl_file_w.write(json.dumps(data) + '\n')
                    save_index+=1
                else:
                    no_save_index += 1
        
        print(save_index)
        print(no_save_index)
