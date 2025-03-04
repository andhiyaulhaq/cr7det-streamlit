import webview
import subprocess


def start_streamlit_app():
    # Run Streamlit in headless mode using subprocess
    subprocess.Popen(
        [
            "streamlit",
            "run",
            "main.py",
            "--server.headless=true",
            "--server.port=8501",
        ]
    )


if __name__ == "__main__":
    # Start the Streamlit app
    start_streamlit_app()

    # Create a pywebview window to display the Streamlit app
    webview.create_window("Streamlit App", "http://127.0.0.1:8501")
    webview.start()
