import requests, base64, json
im_path = "/home/aakash/pic.png"
stream = True
with open(im_path, 'rb') as f:
    image_file = f.read()
    encoded = base64.b64encode(image_file).decode('utf-8')

data = {
    "prompt": '<image>\n say something about this image in 200 words',
    'max_tokens':256,
    'images': [encoded],  # str or a list of str. can be **url** or **base64.**  must match the number of '<image>'
    'stream': stream,
}

res = requests.post(f'http://localhost:8000/generate', json=data)
if stream:
    for line in res.iter_lines():
        if line:
            l = line[6:]
            if l != b'[DONE]':
              print(json.loads(l))
else:
  print(res.json())
