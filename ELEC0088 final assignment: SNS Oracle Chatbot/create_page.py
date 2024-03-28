import tkinter as tk
from tkinter import ttk
from tkinter.font import Font

''' Creating GUI elements using the Tkinter library in Python'''


# Function to center a window on the screen
def center_window(window):
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry('{}x{}+{}+{}'.format(width, height, x, y))


# Function to create page in the window
def create_page(window, title, width, height):
    window.title(title)
    window.geometry(f"{width}x{height}")
    center_window(window)
    window.grid_rowconfigure(0, weight=1)
    window.grid_columnconfigure(0, weight=1)
    frame = ttk.Frame(window)
    frame.grid(row=0, column=0, padx=10, pady=10)
    return frame


# Function to create option in the frame
def create_option(frame, title, row, option):
    option_label = ttk.Label(frame, text=title, font=("Helvetica", 12, "bold"))
    option_label.grid(row=row, column=0, padx=5, pady=5)
    options = option
    selected_option = tk.StringVar()
    option_combobox = ttk.Combobox(frame, textvariable=selected_option, values=options, state="readonly")
    option_combobox.grid(row=row, column=1, padx=5, pady=5)
    option_combobox.current(0)
    return selected_option


# Function to create message label in the frame
def create_message(frame, message, row):
    label = ttk.Label(frame, text=message, font=Font(family="Helvetica", size=20, weight="bold"))
    label.grid(row=row, column=0, columnspan=2, padx=10, pady=5)


# Function to create a smaller message label in the frame
def create_message_small(frame, message, row):
    label = ttk.Label(frame, text=message, font=Font(family="Helvetica", size=12, weight="bold"))
    label.grid(row=row, column=0, columnspan=2, padx=10, pady=5)


# Function to create an entry field in the frame
def create_entry(frame, title, row):
    label = ttk.Label(frame, text=title, font=Font(family="Helvetica", size=12, weight="bold"))
    label.grid(row=row, column=0, padx=10, pady=5)
    entry = ttk.Entry(frame, font=Font(family="Helvetica", size=12, weight="bold"))
    entry.grid(row=row, column=1, padx=10, pady=5)
    return entry


# Function to create a table (Treeview) in the window
def create_table(window, row, data_dict, location_row, width):
    tree = ttk.Treeview(window)

    # Define columns based on the keys of the data dictionary
    columns = tuple(data_dict.keys())
    tree["columns"] = columns
    tree.heading("#0", text="")
    tree.column("#0", width=1)

    # Format columns
    for idx, col in enumerate(columns):
        tree.column(idx, anchor=tk.CENTER, width=width)  # Adjust width as needed

    # Create headings
    for idx, col in enumerate(columns):
        tree.heading(idx, text=col)

    # Add data to the table
    for i in range(row):
        values = [data_dict[col][i] for col in columns]
        tree.insert("", "end", values=values)

    # Place the Treeview widget
    tree.grid(row=location_row, column=0, sticky="nsew")
