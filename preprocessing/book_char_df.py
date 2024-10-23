import pandas as pd
from tqdm import tqdm
from page_raw_dict_reformat import page_raw_dict_reformat
def book_char_df(vol, pages):
    """
    This function takes in a volume number and a list of pages and returns 
    a dataframe of all characters in the book. 
    INPUTS: 
        vol: String corresponding to volume number 
        pages: a list of all page instances in the book. A page is an instance of document  at index page_num
    OUTPUT:
        returns a pandas dataframe with column names:
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
    """
    word_list = []
    for page_num in tqdm(range(len(pages))):
        page = pages[page_num]
        page_raw_dict_reformat(vol, page, page_num, word_list)
    return pd.DataFrame(word_list)