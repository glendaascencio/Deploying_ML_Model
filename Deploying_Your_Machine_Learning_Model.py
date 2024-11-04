# GA 									Nov 4, 2024 
# !pip install litserve
# 𝐥𝐢𝐭𝐬𝐞𝐫𝐯𝐞 is the new kid on the ML deployment block
'''
✅ Open-source
✅ Easy to use
✅ 2x faster than using FastAPI by yourself
✅ Supports Batching and Streaming
✅ GPU Autoscaling
✅ Automatic Dockerization
'''
import joblib, numpy as np 
import litserve as ls

class XGBooastAPI(ls.LitAPI):
	def setup(self, device):
		self.model = joblib.load("model.joblib")

	def decode_request(self, request):
		x = np.asarray(request["input"])
		x = np.expand_dims(x, 0)
		return x 

	def predict(self, x):
		return self.model.predict(x)

	def encode_response(self, output):
		return {"class_idx": int(output)}


if __name__ = "__mian__":
	api = XGBooastAPI()
	server = ls.LitServer(api)
	server.run(port=8000)



























