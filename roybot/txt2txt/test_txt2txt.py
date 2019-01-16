from txt2txt import build_model, infer
model, params = build_model('test/params')
model.load_weights('test/checkpoint')

while(True):
	user_in = input("Justin: ")
	print(infer(user_in.lower(), model, params))


