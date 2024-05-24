import jwt

# Your token
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzE1MDMzMDU1LCJpYXQiOjE3MTUwMjk0NTUsImp0aSI6ImYzNjZjMWZjM2QyOTQ2Mjc5YjM1MGQ2MGM3ZTQ3ZDdiIiwidXNlcl9pZCI6MX0.N67cj8vgQ3ttRUhMREwoMExGLLZeLVl4IJpQwAlBlz4"

# Your secret key
secret_key = "django-insecure-mqdda)7#)9nm(&+*9ko95-kp0o5pg$8mhor-%v63*d5_2rn&ps"

try:
    # Decode the token
    decoded_token = jwt.decode(token, key=secret_key, algorithms=["HS256"])
    
    # Print the decoded token
    print(decoded_token)
except jwt.exceptions.DecodeError as e:
    print("Error decoding token:", e)
except jwt.exceptions.InvalidSignatureError as e:
    print("Invalid signature:", e)
