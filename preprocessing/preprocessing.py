#basically main for this file 
import fitz
import numpy as np
import pandas as pd
from tqdm import tqdm
from book_char_df import book_char_df
from process_words_in_place import process_words_in_place
from rearrange_cols import rearrange_cols

# can make these arguments to type in
# even better have a config file to take care of this! :] 

# ------------------------ IMPORT VOLUMES ------------------------ #
vol1_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 1.pdf'
vol2_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 2.pdf'
vol3_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 3.pdf'

vol1_doc = fitz.open(vol1_path)
vol2_doc = fitz.open(vol2_path)
vol3_doc = fitz.open(vol3_path)

vol1_pages = [vol1_doc[i] for i in range(vol1_doc.page_count)]
vol2_pages = [vol2_doc[i] for i in range(vol2_doc.page_count)]
vol3_pages = [vol3_doc[i] for i in range(vol3_doc.page_count)]
# ---------------------------------------------------------------- #


# ---------------------- Set Global Values ----------------------- #
TARGET_DPI = 300
mat = fitz.Matrix(TARGET_DPI/ 72, TARGET_DPI/ 72)
# ---------------------------------------------------------------- #


print("\nextracting volume 1")
vol1_df = book_char_df("1", vol1_pages)
process_words_in_place(vol1_df)
vol1_df = rearrange_cols(vol1_df)

print("\nextracting volume 2")
vol2_df = book_char_df("2", vol2_pages)
process_words_in_place(vol2_df)
vol2_df = rearrange_cols(vol2_df)

print("\nextracting volume 3")
vol3_df = book_char_df("3", vol3_pages)
process_words_in_place(vol3_df)
vol3_df = rearrange_cols(vol3_df)

print("\n\nSaving volume 1")
vol1_df.to_pickle("../input/char_df/vol1_df.pkl")

print("Saving volume 2")
vol2_df.to_pickle("../input/char_df/vol2_df.pkl")

print("Saving volume 3")
vol3_df.to_pickle("../input/char_df/vol3_df.pkl")