from openai import OpenAI
import httpx
http_client = httpx.Client(verify="./.venv/lib/python3.12/site-packages/certifi/cacert.pem")

client = OpenAI(
    
    http_client = http_client,
    
) 

