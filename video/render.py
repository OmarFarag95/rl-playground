"""Render showcase.html to an MP4, frame by frame.

Needs Playwright, numpy and an ffmpeg with libx264 (pip install playwright numpy imageio-ffmpeg).
The soundtrack is synthesised from the film's own event list (see sound.py) and muxed in at the end.

    python video/render.py                       # video/out/diver-showcase.mp4
    python video/render.py --stills 5,20,40      # only save PNGs at those seconds, to check the look
    python video/render.py --vertical            # 1080×1920 for TikTok / Reels / Shorts
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sound import soundtrack  # noqa: E402

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
    ap.add_argument("--out", default=None)
    ap.add_argument("--vertical", action="store_true", help="9:16 portrait, 1080×1920")
    ap.add_argument("--stills", help="comma-separated seconds; save PNGs instead of a video")
    ap.add_argument("--chrome", default=None, help="path to a Chrome/Chromium binary")
    ap.add_argument("--ffmpeg", default=None)
    a = ap.parse_args()
    a.out = a.out or str(OUT / ("diver-showcase-tiktok.mp4" if a.vertical else "diver-showcase.mp4"))
    size = (1080, 1920) if a.vertical else (1920, 1080)
    srv = serve()
    url = f"http://127.0.0.1:{srv.server_address[1]}/video/showcase.html?capture" + ("&vertical" if a.vertical else "")
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=a.chrome, args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": size[0], "height": size[1]})
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
                path = OUT / f"still{'-v' if a.vertical else ''}-{s:05.1f}.png"
                pg.screenshot(path=str(path))
                print("saved", path)
        else:
            ff = ffmpeg_exe(a.ffmpeg)
            silent, wav = OUT / ".video-only.mp4", OUT / ("soundtrack-tiktok.wav" if a.vertical else "soundtrack.wav")
            enc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(a.fps),
                                    "-c:v", "png", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                                    "-preset", "slow", str(silent)], stdin=subprocess.PIPE)
            t0 = time.time()
            for f in range(frames):
                pg.evaluate(f"window.__step({dt})")
                enc.stdin.write(pg.screenshot(type="png"))
                if f % (a.fps * 5) == 0:
                    print(f"frame {f}/{frames}  {time.time() - t0:.0f}s", flush=True)
            enc.stdin.close()
            enc.wait()
            events = pg.evaluate("window.__events")
            soundtrack(events, frames / a.fps, wav)
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav), "-c:v", "copy", "-c:a", "aac",
                            "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.out], check=True)
            silent.unlink()
            print(f"wrote {a.out}: {duration:.1f} s at {a.fps} fps, {sum(e['kind'] == 'splash' for e in events)} splashes in the soundtrack")
        if errors:
            print("page errors:", *errors, sep="\n  ")
        b.close()
    srv.shutdown()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
