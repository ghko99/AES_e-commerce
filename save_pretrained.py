from transformers import TFAutoModel , AutoTokenizer


model = TFAutoModel.from_pretrained('./model/tfmodel.h5')
tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')