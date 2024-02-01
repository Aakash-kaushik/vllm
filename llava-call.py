import requests, base64, json
im_path = "/home/aakash/pic.png"
stream = True
with open(im_path, 'rb') as f:
    image_file = f.read()
    encoded = base64.b64encode(image_file).decode('utf-8')
headers = {'Content-type': 'application/json'}
# data = {
#     "prompt": '<image>\n say something about this image in 200 words',
#     'max_tokens':256,
#     'images': [encoded],  # str or a list of str. can be **url** or **base64.**  must match the number of '<image>'
#     'stream': stream,
# }

data = {
    "model": "llava-hf/llava-1.5-7b-hf",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What’s in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            }
          }
        ]
      }
    ],
    "stream": stream,
    "max_tokens": 300
  }

res = requests.post(f'http://localhost:8000/v1/chat/completions', json=data, headers=headers)
if stream:
    for line in res.iter_lines():
        if line:
            l = line[6:]
            if l != b'[DONE]':
              print(json.loads(l))
else:
  print(res.json())
