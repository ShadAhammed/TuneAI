"""File selection dialog for loading training data.

SelectData is intentionally simple: it wraps a native OS file-picker so
the user can point the tool at any Excel file without editing code.
The complication/dataset name is optional metadata used for labelling
output plots and summary tables.
"""

import sys
import tkinter as tk
from tkinter.filedialog import askopenfilename

import pandas as pd


class SelectData:
    """Prompt the user to pick an Excel file and load it as a DataFrame.

    Parameters
    ----------
    label:
        A short human-readable name for the dataset (e.g. 'Cardiovascular').
        Used only for console messages and plot titles.
    """

    def __init__(self, label: str = 'Dataset') -> None:
        self.label = label
        print(f'\nDataset: {label}\n')

    def select_file(self) -> pd.DataFrame:
        """Open an OS file-picker dialog and return the selected file as a DataFrame.

        The dialog restricts selection to Excel files.  If the user cancels
        without picking a file the program exits cleanly rather than crashing
        with a confusing error downstream.
        """
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        filepath = askopenfilename(
            title=f'Select data file for: {self.label}',
            filetypes=[
                ('Excel files', '*.xlsx *.xls'),
                ('All files', '*.*'),
            ],
        )
        root.destroy()

        if not filepath:
            print('No file selected — exiting.')
            sys.exit(0)

        try:
            data = pd.read_excel(filepath, index_col=0)
        except Exception as exc:
            raise IOError(
                f"Could not read '{filepath}' as an Excel file: {exc}"
            ) from exc

        print(f"Loaded '{filepath}'\n")
        return data
