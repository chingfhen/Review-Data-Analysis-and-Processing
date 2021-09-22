


from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import numpy as np

model_path = r"C:\Users\tanch\Documents\NTU\NTU Year 3\Sem 1\CZ4045 Natural Language Processing\Assignment 1\local\code\test-squad\checkpoint-96"
model = AutoModelForQuestionAnswering.from_pretrained(model_path)		
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")			

def predict_negation_span(sentence):
    input_ids = tokenizer("negation",
                          sentence, 
                          max_length=30, 
                          padding="max_length", 
                          truncation="only_second",
                          return_tensors='pt')['input_ids']
    pred = model(input_ids)
    start_position  =np.argmax(pred.start_logits[0].detach().numpy())
    end_position  =np.argmax(pred.end_logits[0].detach().numpy())
    if start_position==0 or end_position==0:
        return "<no_answer>"
    return tokenizer.decode(input_ids[0][start_position: end_position+1]).strip()		