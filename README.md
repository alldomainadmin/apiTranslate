# Translate AI by API
Translate AI system by API

## Command line
* Description `tutorial Ubuntu`
```
cd ~/Python/ && python3.12 -m venv envAPI
source ~/Python/envAPI/bin/activate
# pip3 install fastapi
# pip3 install uvicorn
# pip3 install huggingface_hub
# pip3 install pydantic
# pip3 install python-dotenv
pip3 install fastapi uvicorn huggingface_hub pydantic python-dotenv
pip3 install --upgrade pip
pip3 install requests
deactivate

# ~/Python/envAPI/bin/python3.12 main.py
# cd /home/owner/Python/apiTranslate && /home/owner/Python/envAPI/bin/python3.12 /home/owner/Python/apiTranslate/main.py
source /home/owner/Python/envAPI/bin/activate && cd /home/owner/Python/apiTranslate/ && uvicorn main:app --reload --host 0.0.0.0 --port 8000
lsof -i :8000
kill -9 364768 364771

```
