data_path=$1
python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path /path/FlagEmbedding/Chuxin-Embedding \
--input_file ${data_path} \
--output_file ${data_path}_20_200_embminedHNtest.jsonl \
--range_for_sampling 20-200 \
--negative_number 15 \
--query_instruction_for_retrieval "为这个句子生成表示以用于检索相关文章："

