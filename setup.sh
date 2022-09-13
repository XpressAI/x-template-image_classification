if [[ "$OSTYPE" =~ ^msys ]]; then 
    OS=Scripts
else
    OS=bin
fi &&
python -m pip install --upgrade pip &&
python -m venv venv &&
source venv/$OS/activate &&
pip install -r requirements.txt