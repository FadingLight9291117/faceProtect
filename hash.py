import hashlib

m = hashlib.sha256()
s = 'hello world'
m.update(s.encode('utf-8'))
print(m.hexdigest())
