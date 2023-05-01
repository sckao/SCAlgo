# Main Control - Top Level Program Entry Point
# Author: Kevin Kao
# Contact: kaoshihchuan@gmail.com

import tkinter as tk
import SCAlgo.test.TestPanel as TestPanel
import pathlib


def main():

    # Create UI to run
    rootgui = tk.Tk()
    rootgui.title('Ramp_and_Roll_Simulator')

    init_width = int(rootgui.winfo_screenwidth() * 0.9)
    init_height = int(rootgui.winfo_screenheight() * 1.0)
    init_x = int(rootgui.winfo_screenwidth() / 8)
    init_y = int(rootgui.winfo_screenheight() / 8)
    rootgui.geometry('{}x{}+{}+{}'.format(init_width, init_height, init_x, init_y))
    # Create Content for Main UI: rootgui
    TestPanel.GuiWindow(rootgui)

    rootgui.mainloop()

# end main


if __name__ == '__main__':
    main()
# end if
