######
INSTALL

On Windows:

```
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
mkdir code && cd code
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
