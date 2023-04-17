
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"/home/julien/Documents/M1/ArtStudio/testArtStudio/build/assets/frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("1200x800")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 1024,
    width = 1440,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_text(
    20.0,
    20.0,
    anchor="nw",
    text="Tell us an interesting story about the content of these images. Every time a new image appears, try to include it as quickly as possible!",
    fill="#000000",
    font=("Inter", 32 * -1)
)

canvas.create_rectangle(
    74.0,
    189.0,
    459.0,
    904.0,
    fill="#D9D9D9",
    outline="")

canvas.create_rectangle(
    998.0,
    189.0,
    1383.0,
    904.0,
    fill="#D9D9D9",
    outline="")

canvas.create_rectangle(
    536.0,
    189.0,
    921.0,
    904.0,
    fill="#D9D9D9",
    outline="")

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    958.0,
    547.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    498.0,
    547.0,
    image=image_image_2
)
window.resizable(False, False)
window.mainloop()
