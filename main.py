import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))


from app.main import *

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", "app/main.py"]
    sys.exit(stcli.main()) 