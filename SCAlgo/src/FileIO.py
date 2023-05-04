import pandas as pd
import tkinter.filedialog as tkfd
# import typing


class FileIO:

    def __init__(self):
        self.is_file_open: bool = False
        self.file_type_list = ['csv', 'xlsx', 'txt']

    def prompt_to_select_open_file(self, file_type: str) -> str:

        file_type_str = '*.*'
        for value in self.file_type_list:
            if file_type == value:
                file_type_str = '*.' + file_type
                break

        filename = tkfd.askopenfilename(
            initialdir="../data/", title="Select file",
            filetypes=(
                (file_type+" files", file_type_str),
                ("all files", "*.*")
            )
        )
        the_file_name = filename + ('' if filename.endswith('.'+file_type) else '.'+file_type)
        print('Selected %s file: %s! ' % (file_type, the_file_name))
        return the_file_name
    # end prompt_to select_open_file

    @staticmethod
    def open_csv_file1(filename: str, **kwargs):

        data_frame = pd.read_csv(filename, **kwargs)
        print(data_frame)
        np_array = data_frame.to_numpy()
        print(' ====================== ')
        print(np_array)

        return data_frame

    @staticmethod
    def open_excel_file(filename: str, **kwargs):
        data_frame = pd.read_excel(filename, **kwargs)
        return data_frame


f = FileIO()
csv_file_name = f.prompt_to_select_open_file('csv')
f.open_csv_file1(csv_file_name, sep=',', header=3, usecols=[0, 1, 2, 3])
