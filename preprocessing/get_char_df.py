import fitz
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# ------------------------ IMPORT VOLUMES ------------------------ #
vol1_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 1.pdf'
vol2_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 2.pdf'
vol2_last8_path = '../input/vol2_last_8_pages.pdf'
vol3_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 3.pdf'

vol1_doc = fitz.open(vol1_path)
vol2_doc = fitz.open(vol2_path)
vol2_last8_doc = fitz.open(vol2_last8_path)
vol3_doc = fitz.open(vol3_path)

# append vol2_last8 to vol2
vol2_doc.insert_pdf(vol2_last8_doc, from_page=0, to_page=vol2_last8_doc.page_count - 1, start_at=vol2_doc.page_count) 
vol2_doc.save('../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 2 COMPLETE.pdf')


vol1_pages = [vol1_doc[i] for i in range(vol1_doc.page_count)]
vol2_pages = [vol2_doc[i] for i in range(vol2_doc.page_count)]
vol2_last8_pages = [vol2_last8_doc[i] for i in range(vol2_last8_doc.page_count)]
vol3_pages = [vol3_doc[i] for i in range(vol3_doc.page_count)]
# ---------------------------------------------------------------- #

# save pkl file as csv for vol2
# vol2_df = pd.read_pickle("../input/char_df/vol2_df.pkl")
# vol2_df.to_csv("../input/char_df/vol2_df.csv", index = False)
# ---------------------- Set Global Values ----------------------- #
TARGET_DPI = 300
mat = fitz.Matrix(TARGET_DPI/ 72, TARGET_DPI/ 72)
# ---------------------------------------------------------------- #


# ------- Functions For Extracting Each CHARACTER and WORD ------- #
#note bbox not adjusted with TARGET_DPI
def page_raw_dict_reformat(vol, page, page_num, word_list):
    """ DOCUMENTATION
    This function appends the characters and relevent information 
    corresponds to it in each page to word_list. Formatted such that 
    word_list is ready to represent each char as a pandas dataframe row 
    with column names:
        ['vol_num', 
         'page_num', 
         'block_num',
         'block_num_absolute', 
         'block_bbox',
         'line_num', 
         'line_wmode', 
         'line_dir', 
         'line_bbox', 
         'span_num',
         'span_size',
         'span_flags', 
         'span_font', 
         'span_color', 
         'span_ascender',
         'span_descender', 
         'span_origin', 
         'span_bbox', 
         'word_num', 
         'word',
         'char_num', 
         'char', 
         'char_origin', 
         'char_bbox']

    INPUTS: 
        vol: String corresponding to volume number 
        page: instance of document at index page_num
        page_num: Int corresponging to page number
        word_list: List to which each character will be appended
    """
    text_blocks = [
        block for block in page.get_text("rawdict")['blocks'] 
        if block['type'] == 0
        ]

    #for each block in text blocks
    for b_i in range(len(text_blocks)):
        b = text_blocks[b_i]
        curr_block_num_abs = b['number']                    #true blcok number
        curr_block_num_reletive = b_i                       #excludes image blocks
        curr_block_bbox = b['bbox']

        for l_i in range(len(b['lines'])):
            l = b['lines'][l_i]
            curr_line_num = l_i
            curr_line_wmode = l['wmode']
            curr_line_dir = l['dir']
            curr_line_bbox = l['bbox']
            
            word_num = 0
            for s_i in range(len(l['spans'])):
                s = l['spans'][s_i]
                span_num = s_i
                span_size = s['size']
                span_flags = s['flags']
                span_font = s['font']
                span_color = s['color']
                span_ascender = s['ascender']
                span_descender = s['descender'] 
                span_chars = s['chars'] 
                span_origin = s['origin'] 
                span_bbox = s['bbox']
                
                span_word = ""
                char_in_words = []
                
                for span_char_i in range(len(span_chars)):
                    c = span_chars[span_char_i]
                    char = c['c']

                    # if we are at the last character of span or a white space character
                    # add word to dictionary
                    if (span_char_i == len(span_chars) - 1) or \
                        (char.isspace() and len(span_word) > 0): 

                        if (span_char_i == len(span_chars) - 1) and (not char.isspace()): 
                            #to ensure the last character in the span is added (if it's not a space)
                            span_word += char
                            char_in_words.append(c)

                        for c_i in range(len(char_in_words)):
                            char_num = c_i 
                            char_origin = char_in_words[c_i]['origin']
                            char_bbox = char_in_words[c_i]['bbox']
                            curr_char = char_in_words[c_i]['c']

                            char_row = {
                                'vol_num': vol,
                                'page_num': page_num,
                                'block_num': curr_block_num_reletive,
                                'block_num_absolute': curr_block_num_abs,
                                'block_bbox': curr_block_bbox,

                                'line_num': curr_line_num,
                                'line_wmode': curr_line_wmode,
                                'line_dir': curr_line_dir,
                                'line_bbox': curr_line_bbox,
                                
                                'span_num': span_num,
                                'span_size': span_size,
                                'span_flags': span_flags,
                                'span_font': span_font,
                                'span_color': span_color,
                                'span_ascender': span_ascender,
                                'span_descender': span_descender,
                                'span_origin': span_origin,
                                'span_bbox': span_bbox,

                                'word_num': word_num, #in the entire line
                                'word': span_word,

                                'char_num': c_i,
                                'char': curr_char,
                                'char_origin': char_origin,
                                'char_bbox': char_bbox
                            }
                            word_list.append(char_row)

                        word_num += 1
                        span_word = ''
                        char_in_words = []
                    elif not char.isspace():  
                        span_word += char
                        char_in_words.append(c)
                    #only other possibility is that is it a white space in which case we can ignore it

def book_char_df(vol, pages):
    word_list = []
    for page_num in tqdm(range(len(pages))):
        page = pages[page_num]
        page_raw_dict_reformat(vol, page, page_num, word_list)
    return pd.DataFrame(word_list)
# ---------------------------------------------------------------- #


# ------------------------- Process words ------------------------ #
def process_words_in_place(vol_df, drop_coords = True):
    #get word_bbox, prune_word -> word after pruning non-alphanumeric characters in a word (affects word_bbox)
    print("get alphanumeric part of words ... ")
    alnum_word = lambda word : ''.join(c for c in word if c.isalnum())
    vol_df["pruned_word"] = vol_df["word"].apply(alnum_word)

    print("setting word and character coordinates ...")
    coords_str = ["char_x0", "char_y0", "char_x1", "char_y1"]
    for i in range(len(coords_str)):
        vol_df[coords_str[i]] = vol_df["char_bbox"].apply(lambda x: x[i])
    
    non_alnum_coord_toNaN = lambda r, col_result: r[col_result] if r["char"].isalnum() else np.NaN 
    vol_df["pruned_char_x0"] = vol_df.apply(lambda r : non_alnum_coord_toNaN(r, "char_x0"), axis = 1)
    vol_df["pruned_char_y0"] = vol_df.apply(lambda r : non_alnum_coord_toNaN(r, "char_y0"), axis = 1)
    vol_df["pruned_char_x1"] = vol_df.apply(lambda r : non_alnum_coord_toNaN(r, "char_x1"), axis = 1)
    vol_df["pruned_char_y1"] = vol_df.apply(lambda r : non_alnum_coord_toNaN(r, "char_y1"), axis = 1)

    group_cols = vol_df.columns.difference(["char_num", "char", "char_origin", "char_bbox", "char_x0", "char_y0", "char_x1", "char_y1", "pruned_char_x0", "pruned_char_y0", "pruned_char_x1", "pruned_char_y1"], sort=False).tolist()
    print("grouping the words ...")
    #for each word get the coordinates of boundary box
    vol_df["word_x0"] = vol_df.groupby(group_cols)['char_x0'].transform('min')
    vol_df["word_y0"] = vol_df.groupby(group_cols)['char_y0'].transform('min')
    vol_df["word_x1"] = vol_df.groupby(group_cols)['char_x1'].transform('max')
    vol_df["word_y1"] = vol_df.groupby(group_cols)['char_y1'].transform('max')

    vol_df["pruned_word_x0"] = vol_df.groupby(group_cols)['pruned_char_x0'].transform('min')
    vol_df["pruned_word_y0"] = vol_df.groupby(group_cols)['pruned_char_y0'].transform('min')
    vol_df["pruned_word_x1"] = vol_df.groupby(group_cols)['pruned_char_x1'].transform('max')
    vol_df["pruned_word_y1"] = vol_df.groupby(group_cols)['pruned_char_y1'].transform('max')

    print("getting bounding box tuples ...")
    #from single coords to bbox tuples
    vol_df["word_bbox"] = vol_df.apply(lambda r: (r["word_x0"], r["word_y0"], r["word_x1"], r["word_y1"]), axis = 1)
    vol_df["pruned_word_bbox"] = vol_df.apply(lambda r: (r["pruned_word_x0"], r["pruned_word_y0"], r["pruned_word_x1"], r["pruned_word_y1"]), axis = 1)

    if drop_coords:
        print("dropping single coordinate columns ...")
        vol_df.drop(columns= ["char_x0", "char_y0", "char_x1", "char_y1", 
                              "word_x0", "word_y0", "word_x1", "word_y1",
                              "pruned_char_x0", "pruned_char_y0", "pruned_char_x1", "pruned_char_y1",
                              "pruned_word_x0", "pruned_word_y0", "pruned_word_x1", "pruned_word_y1"
                             ], inplace = True)

# ---------------------------------------------------------------- #


# ----------------------- Order columns ---------------------- #
# only useful when seeing the dataframe visually and is not necessary
#      so can be removed
def rearrange_cols(vol_df): 
    vol_based =   [c for c in vol_df.columns if c.startswith("vol")]
    page_based =  [c for c in vol_df.columns if c.startswith("page")]
    block_based = [c for c in vol_df.columns if c.startswith("block")]
    line_based =  [c for c in vol_df.columns if c.startswith("line")]
    span_based =  [c for c in vol_df.columns if c.startswith("span")]
    word_based =  [c for c in vol_df.columns if c.startswith("word")]
    prune_based = [c for c in vol_df.columns if c.startswith("pruned")]
    char_based =  [c for c in vol_df.columns if c.startswith("char")]

    new_cols = vol_based + \
               page_based + \
               block_based + \
               line_based + \
               span_based + \
               word_based + \
               prune_based + \
               char_based 
    if len(new_cols) == len(vol_df.columns): 
        print("columns successfully rearranged")
        return vol_df[new_cols]
    else: 
        print("**WARNING** \n \t columns not rearranged")
        return vol_df
# ---------------------------------------------------------------- #


# ----------------------- Extract And Save ----------------------- #
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

if not os.path.exists("../input/char_df"):
    os.makedirs("../input/char_df")

print("\n\nSaving volume 1")
vol1_df.to_pickle("../input/char_df/vol1_df.pkl")

print("Saving volume 2")
vol2_df.to_pickle("../input/char_df/vol2_df.pkl")

print("Saving volume 3")
vol3_df.to_pickle("../input/char_df/vol3_df.pkl")
# ---------------------------------------------------------------- #