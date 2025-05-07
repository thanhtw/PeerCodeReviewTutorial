import os
import streamlit.web.cli as stcli
import sys

if __name__ == "__main__":
    # Define default port or use environment variable
    port = int(os.environ.get("PORT", 8505))
    
    # Set the host to 0.0.0.0 to make it accessible from any IP
    sys.argv = ["streamlit", "run", "app.py", "--server.port", str(port), "--server.address", "0.0.0.0"]
    sys.exit(stcli.main())