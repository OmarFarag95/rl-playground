"""Render showcase.html to an MP4, frame by frame.

Needs Playwright and an ffmpeg with libx264 (pip install playwright imageio-ffmpeg).

    python video/render.py                       # video/out/diver-showcase.mp4
    python video/render.py --stills 5,20,40      # only save PNGs at those seconds, to check the look
"""
import argparse
import functools
import http.server
import subprocess
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "video" / "out"


def serve():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    handler = functools.partial(Quiet, directory=str(ROOT))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def ffmpeg_exe(given):
    if given:
        return given
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=str(OUT / "diver-showcase.mp4"))
    ap.add_argument("--stills", help="comma-separated seconds; save PNGs instead of a video")
    ap.add_argument("--chrome", default=None, help="path to a Chrome/Chromium binary")
    ap.add_argument("--ffmpeg", default=None)
    a = ap.parse_args()
    srv = serve()
    url = f"http://127.0.0.1:{srv.server_address[1]}/video/showcase.html?capture"
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=a.chrome, args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 1920, "height": 1080})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(url)
        pg.wait_for_function("window.__ready === true", timeout=60000)
        duration = pg.evaluate("window.__duration")
        dt = 1 / a.fps
        frames = int(duration * a.fps)
        if a.stills:
            want = sorted(float(s) for s in a.stills.split(","))
            t = 0.0
            for s in want:
                while t + dt <= s:
                    pg.evaluate(f"window.__step({dt})")
                    t += dt
                path = OUT / f"still-{s:05.1f}.png"
                pg.screenshot(path=str(path))
                print("saved", path)
        else:
            enc = subprocess.Popen([ffmpeg_exe(a.ffmpeg), "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(a.fps),
                                    "-c:v", "png", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                                    "-preset", "slow", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
            t0 = time.time()
            for f in range(frames):
                pg.evaluate(f"window.__step({dt})")
                enc.stdin.write(pg.screenshot(type="png"))
                if f % (a.fps * 5) == 0:
                    print(f"frame {f}/{frames}  {time.time() - t0:.0f}s", flush=True)
            enc.stdin.close()
            enc.wait()
            print(f"wrote {a.out}: {duration:.1f} s at {a.fps} fps")
        if errors:
            print("page errors:", *errors, sep="\n  ")
        b.close()
    srv.shutdown()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
