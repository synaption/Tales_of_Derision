"""Launcher for the ink workbench.

The demo used to be one file. It is now three, split where the seams actually
are so the effect can be lifted into the game on its own:

    inkfx.py     the effect -- parchment, wet ink, the drawing pen. No window,
                 no artwork, no controls. Runs headless.
    inkbench.py  this application: the window, the menus and sliders, the demo
                 page, and the keys that drive them.
    pentrace.py  recovering pen strokes from a finished drawing. Pure arrays.

This file stays so that `python3 inkGL.py` still starts the workbench.
"""

from inkbench import main

if __name__ == "__main__":
    # Only when run as a program: importing this must never relaunch anything,
    # or a test harness that imports it would restart itself.
    main(allow_restart=True)
