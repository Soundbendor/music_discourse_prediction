from transformers.models.distilbert.tokenization_distilbert import DistilBertTokenizer

distil_bert = 'distilbert-base-uncased-emotion'

tokenizer = DistilBertTokenizer.from_pretrained(distil_bert, do_lower_case=True, add_special_tokens=True)