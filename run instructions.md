# Run Instructions

## 1. Install dependencies
```bash
pip3 install -r requirements.txt
```
On macOS with Homebrew Python you may need:
```bash
pip3 install -r requirements.txt --break-system-packages
```

## 2. (Optional) Test the model in the terminal
```bash
cd src
python3 heat_model.py
```

## 3. Run the dashboard
```bash
cd src
python3 app.py
```

Open your browser to: http://127.0.0.1:8050

If port 8050 is already in use:
```bash
lsof -ti:8050 | xargs kill -9
python3 app.py
```

Stop the server with Ctrl+C.
