from transformers import FSMTForConditionalGeneration, FSMTTokenizer
# mname = "facebook/wmt19-en-ru"
# mname = "facebook/wmt19-ru-en"
# mname = "facebook/wmt19-en-de"
mname = "facebook/wmt19-de-en"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

tokenizer.save_pretrained("./saved_model/translation_models/" + mname + "/")
model.save_pretrained("./saved_model/translation_models/" + mname + "/")

input = "Machine learning is great, isn't it?"
input_ids = tokenizer.encode(input, return_tensors="pt")
outputs = model.generate(input_ids)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded) # Машинное обучение - это здорово, не так ли?