import argparse
import os
import webbrowser

import torch
import uvicorn


def main():
    p = argparse.ArgumentParser(description="Splash Lab: a reinforcement-learning playground.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1),
                   help="CPU threads PyTorch may use (small networks train fastest on few threads)")
    p.add_argument("--open", action="store_true", help="open the browser")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    url = f"http://{a.host}:{a.port}"
    print(f"Splash Lab running at {url}")
    if a.open:
        webbrowser.open(url)
    uvicorn.run("rlplay.server:app", host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
