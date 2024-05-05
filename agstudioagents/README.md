######
INSTALL

-------
CRITICAL WARNING

The below will enable an AI model to generate and run code on your computer. Do not do this unless YOU TRUST THE AI MODEL to not write malicious code.


On Windows:
```
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
mkdir code && cd code
git clone git@github.com:ruddythor/mick.ai.git
cd mick.ai/agstudioagents
python3 -m venv ./venvy
.\venvy\Scripts\Activate.ps1
pip install -r reqs.txt
```

######
RUN

On Windows:

Install LM Studio and run Llama3 dolphin uncensored 8b in server mode on localhost:8000/v1.


Then:

```
python main.py
```
