import unicodedata
import json
import sys
import os
import re
from difflib import SequenceMatcher


# 正则表达式匹配目录关键字
def is_directory(s):
    
    pattern1 = r'第(\d+)回'
    pattern2 = r'第(\d+)百章'
    pattern3 = r'目录\s+(第\d+章\s+)?(第\d+节\s+)'
    pattern4 = r'概述\s+(第\d+章\s+)?(第\d+节\s+)'
    if re.search(pattern1, s) or re.search(pattern2, s) or re.search(pattern3, s) or re.search(pattern4, s):
        return True
    else:
        return False

# 正则表达式匹配时效关键字
pattern = r'现在|立刻|马上|目前|此刻|当前|过去|以前|从前|往昔|昔日|旧时|将来|未来|以后|随后|接下来|往后|当前|即将|目前|今|最近|近日|近期|昨' 


symbol_list = "�□★〓＼㊣℡﹋﹌☆★○●◎◇◆□■▓△▲▼▽◢◣◤◥⊿※§々¤￠▶▷■▌█◀►▍◾▊▎▉▅▇▋◼┃"
# 匹配特殊字符
pattern_spe_signal = re.compile(r'[^-_a-zA-Zāáǎàēéěèīíǐìōóǒòūúǔùü\d\u4e00-\u9fa5\s,<.>/?;:"\[{\]}|`~!@#$%^&*()=+，ニ〇〜〚〛▪《。》？λ／｜；：＊ε＝‘’“”【】、·！￥…（）—×≥≤＜¥〈〉μαω•＋±「」–─〔〕—『』∶﹔∕Ω―•°β．－～℃％㎡√/÷.Ⅱ﹝﹞㎥﹪㎜㎎㎏㎝>━²³—﹢πⅤ≈γ≠‖£⇌Φδ≧║≦, Øσ]')

def _is_replace(char):
    """Checks whether `chars` is a control character."""
    # skip the following control chars
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C") or cat.startswith("P") or "Sk" in cat or "So" in cat or "Lo" in cat or "Mn" in cat or "Mc" in cat or "Me" in cat:
        return True
    return False

def replace_char(char_list, text):
    """
    remove control chars
    """
    for idx, char in enumerate(char_list):
        if _is_replace(char):
            text = text.replace(char, "")
    return text

def remove_emoji(text):
    regex_msg = re.compile(
        '[\U0001F600-\U0001F92F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F190-\U0001F1FF\U00002702-\U000027B0\U0001F926-\U0001FA9F\u200d\u2640-\u2642\u2600-\u2B55\u23cf\u23e9\u231a\ufe0f\u3000\uf033\u23ee\ue50b\ue50a\U0000200D\xa0\u2002' + ']+')
    text_ = regex_msg.sub(u'', text)
    text_ = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text_)
    text_ = re.sub(r'\\u[0-9a-fA-F]{4}', '', text_)
    text_ = re.sub(r'\\\\u[0-9a-fA-F]{4}', '', text_)
    text_ = re.sub(r'u[0-9a-fA-F]{4}', '', text_)
    text_ = text_.replace("<br>", "").replace("===", "").replace("\u3000", "")
    return  text_

def preprocess_text(text):
    # serch所有非中英文正常字符replace
    
    match = re.findall(pattern_spe_signal, text)
    
    text = remove_emoji(replace_char(match, text))
    for sym in symbol_list:
        text = text.replace(sym, "")
    return text


# 指定你的JSONL文件路径
data_path = sys.argv[1]
if os.path.isdir(data_path):
    file_path_r_list = os.listdir(data_path)
    file_path_r_list = [i for i in file_path_r_list if "json" in i]
    file_path_r_list = [os.path.join(data_path,i) for i in file_path_r_list]
else:
    file_path_r_list = [data_path]


for i in file_path_r_list:

    filew = open(f"{i}_rulespairs.json", 'w')
    # 打开并读取JSONL文件
    index = 0
    index_save = 0
    
    with open(i, 'r', encoding='utf-8') as file:
        # 逐行读取文件
    
        for line in file:
            index += 1
            datas = json.loads(line)
            
            # query 具有时效性，比如今天/昨天等则删除。
            #timeflag = re.findall(pattern, datas["query"])
            #if len(timeflag)>0:
            #   continue
            
            # # LLM 生成结果简单清洗
            datas["query"] = datas["query"].replace("\"","")
            datas["query"] = datas["query"].split("\n")[0]
            if "user_query" in datas["query"]:
                datas["query"] = datas["query"].split("user_query")[-1]
                datas["query"] = datas["query"].strip(": \"")

            if "positive_document" in datas["query"]:
                datas["query"] = datas["query"].split("positive_document")[0]
            

             # 清洗特殊字符
            
            datas['pos'] = [preprocess_text(text) for text in datas['pos'] if text != ""]
            datas['neg'] = [preprocess_text(text) for text in datas['neg'] if text != ""]
         
            # pos 和 neg 内容是否重复简单，比如章节目录
            try: 
              datas["pos"] = [x for x in datas["pos"] if not is_directory(x)]
              datas["neg"] = [x for x in datas["neg"] if not is_directory(x)]
            except:
              pass
            try:
              datas["pos_scores"] = [x for x,y in zip(datas["pos_scores"], datas["pos"]) if not is_directory(y)]
              datas["neg_scores"] = [x for x,y in zip(datas["neg_scores"], datas["neg"]) if not is_directory(y)]
            except:
              pass
            
            # 计算 （query，pos）（query，neg）length长度，删除pos 短于 query 1/2的
            datas["pos"] = [x for x, y in zip(datas["pos"], datas["pos"]) if len(datas["query"]) / (len(y)+0.1) < 2]
            datas["neg"] = [x for x, y in zip(datas["neg"], datas["neg"]) if len(datas["query"]) / (len(y)+0.1) < 2]
            try:
              datas["pos_scores"] = [x for x, y in zip(datas["pos_scores"], datas["pos"]) if len(datas["query"]) / (len(y)+0.1) < 2]
              datas["neg_scores"] = [x for x, y in zip(datas["neg_scores"], datas["neg"]) if len(datas["query"]) / (len(y)+0.1) < 2]
            except:
              pass

            # 计算 （query，pos）（query，neg）相似度， 删除相似度高于0.5的
            zh_chars = re.findall(r'[\u4e00-\u9fa5]', datas["query"])
            try:
              zh_percent = len(zh_chars) / len(datas["query"])
            except:
              zh_percent = 0
            if zh_percent > 0.1:
              pos_similarity = [SequenceMatcher(None, datas["query"], i).ratio() for i in datas["pos"]]
              neg_similarity = [SequenceMatcher(None, datas["query"], i).ratio() for i in datas["neg"]]
              datas["pos"] = [x for x, y in zip(datas["pos"], pos_similarity) if y < 0.5]
              datas["neg"] = [x for x, y in zip(datas["neg"], neg_similarity) if y < 0.5]
              try:
                  datas["pos_scores"] = [x for x, y in zip(datas["pos_scores"], pos_similarity) if y < 0.5]
                  datas["neg_scores"] = [x for x, y in zip(datas["neg_scores"], neg_similarity) if y < 0.5]
              except:
                  pass
            
            try:
                if len(datas["pos"]) > 0 and len(datas["neg"]) > 0 and len(datas["query"]) > 1 and len(datas["query"]) < 64:
                    index_save += 1
                    filew.write(json.dumps(datas) + '\n')
                

            except:
                
                
                if len(datas["pos"]) > 0 and len(datas["query"]) > 1 and len(datas["query"]) < 64:
                    index_save += 1
                    filew.write(json.dumps(datas) + '\n')
    print(index)
    print(index_save)
