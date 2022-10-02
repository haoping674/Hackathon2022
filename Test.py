#%%
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
#%%
ner_tokenizer = AutoTokenizer.from_pretrained("dominiqueblok/roberta-base-finetuned-ner")
ner_model = AutoModelForTokenClassification.from_pretrained("dominiqueblok/roberta-base-finetuned-ner")

embedding_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
embedding_model = AutoModel.from_pretrained("xlm-roberta-base")
#%%
test_title = "Creep And Creep Rupture Of Strongly Reinforced Metallic Composites"
test_keyword = "Metal"
#%%
ner_token = ner_tokenizer.encode(test_title)
print(ner_token)
# %%
