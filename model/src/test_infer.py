import requests, json
url = "http://127.0.0.1:5000/infer"
files = {"image": open("D:/civic-issue-ai/data/pilot/images/potholes/Image_2.jpg","rb")}
data = {"text":"big pothole near crosswalk","lat":"28.7041","lon":"77.1025"}
r = requests.post(url, files=files, data=data, timeout=20)
print("status:", r.status_code)
print(json.dumps(r.json(), indent=2))
