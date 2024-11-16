import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
from cProfile import label #?not sure
import re
#from fuzzywuzzy import fuzz
import difflib 
#from fuzzywuzzy import process
import time
from tqdm import tqdm 
import fitz
from fitz.utils import getColor

from functools import reduce
import os 
tqdm.pandas()

# ------------------------ IMPORT VOLUMES ------------------------ #
vol1_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 1.pdf'
vol2_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 2 COMPLETE.pdf'
vol3_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 3.pdf'

vol1_doc = fitz.open(vol1_path)
vol2_doc = fitz.open(vol2_path)
vol3_doc = fitz.open(vol3_path)

vol1_pages = [vol1_doc[i] for i in range(vol1_doc.page_count)]
vol2_pages = [vol2_doc[i] for i in range(vol2_doc.page_count)]
vol3_pages = [vol3_doc[i] for i in range(vol3_doc.page_count)]

vol1_char_df = pd.read_pickle("../input/char_df/vol1_df.pkl")
vol2_char_df = pd.read_pickle("../input/char_df/vol2_df.pkl")
vol3_char_df = pd.read_pickle("../input/char_df/vol3_df.pkl")
# ---------------------------------------------------------------- #

# ---------------------- char_df -> word_df ---------------------- #
vol1_word_df = vol1_char_df.loc[:, ['vol_num', 'page_num', 
                                    'block_num', 'block_num_absolute', 'block_bbox',
                                    'line_num', 'line_wmode', 'line_dir', 'line_bbox', 
                                    'span_num', 'span_size', 'span_flags', 'span_font', 'span_color', 'span_ascender', 'span_descender', 'span_origin', 'span_bbox', 
                                    'word_num', 'word','word_bbox', 'pruned_word', 'pruned_word_bbox']].drop_duplicates()

vol2_word_df = vol2_char_df.loc[:, ['vol_num', 'page_num', 
                                    'block_num', 'block_num_absolute', 'block_bbox',
                                    'line_num', 'line_wmode', 'line_dir', 'line_bbox', 
                                    'span_num', 'span_size', 'span_flags', 'span_font', 'span_color', 'span_ascender', 'span_descender', 'span_origin', 'span_bbox', 
                                    'word_num', 'word','word_bbox', 'pruned_word', 'pruned_word_bbox']].drop_duplicates()

vol3_word_df = vol3_char_df.loc[:, ['vol_num', 'page_num', 
                                    'block_num', 'block_num_absolute', 'block_bbox',
                                    'line_num', 'line_wmode', 'line_dir', 'line_bbox', 
                                    'span_num', 'span_size', 'span_flags', 'span_font', 'span_color', 'span_ascender', 'span_descender', 'span_origin', 'span_bbox', 
                                    'word_num', 'word','word_bbox', 'pruned_word', 'pruned_word_bbox']].drop_duplicates()
# ---------------------------------------------------------------- #

# ------------------------- IMPORT INDEX ------------------------- #
vol1_index_path = '../output/local/index_output/vol1_nonitalics.csv'
vol2_index_path = '../output/local/index_output/vol2_nonitalics.csv'
vol3_index_path = '../output/local/index_output/vol3_nonitalics.csv'

vol1_index_df = pd.read_csv(vol1_index_path)
vol2_index_df = pd.read_csv(vol2_index_path)
vol3_index_df = pd.read_csv(vol3_index_path)

#changing name of columns of mout. indecies 
vol1_index_df.rename(columns={'closest_genus': 'mouterde_genus', 'closest_epithet': 'mouterde_epithet', 'authors':'mouterde_author', 'closest_infra_name':'mouterde_infra'}, inplace=True)
vol2_index_df.rename(columns={'closest_genus': 'mouterde_genus', 'closest_epithet': 'mouterde_epithet', 'authors':'mouterde_author', 'closest_infra_name':'mouterde_infra'}, inplace=True)
vol3_index_df.rename(columns={'closest_genus': 'mouterde_genus', 'closest_epithet': 'mouterde_epithet', 'authors':'mouterde_author', 'closest_infra_name':'mouterde_infra'}, inplace=True)

vol_index_lists = []
#list of genera from index -- uppercased to match main text pattern
i = 1 
for vol_index_df in [vol1_index_df, vol2_index_df, vol3_index_df]:
    print("gathering index genera and species", str(i))
    vol_genera = vol_index_df[vol_index_df['taxon_rank'] == 'genus']['mouterde_genus'].str.upper().tolist()
    #list of species binomial from index
    vol_species_temp_df = vol_index_df[(vol_index_df['taxon_rank'] == 'epithet') & (~vol_index_df['mouterde_genus'].isna())]
    vol_species_binomial_list = list(zip(vol_species_temp_df['mouterde_genus'], vol_species_temp_df['mouterde_epithet']))
    vol_species_abriviation = list(map(lambda x: f"{x[0][0]}. {x[1]}", vol_species_binomial_list))
    vol_species = list(map(lambda x: f"{x[0]} {x[1]}", vol_species_binomial_list))

    vol_index_lists.append([vol_genera, vol_species, vol_species_abriviation])
    i += 1 
# ---------------------------------------------------------------- #

# --------------------- processing functions --------------------- #
def is_italic(flags):
    """ DOCUMENTATION 
    given fitz flag, checks if it's italics 
    INPUTS:
        flag: Int indicating the flag
    OUTPUT:
        Boolean value (True if flag indicates italics; False othervise)
    """
    return flags & 2 ** 1 != 0


def get_n_words_flagged(df, n, inplace = True):
    """ DOCUMENTATION 
    given a dataframe of words in the book, and number n, for each word (w0) in 
    each line (l0), creates starting of n words starting from w0 (n_word_string). If there are 
    k < n words left on the line (l0), the k word string starting from w0 is used instead. 
    Additionally, the function also finds the union (OR) of the flags of the words in
    each n_word_string. 
    INPUTS:
        df: dataframe containing the words in the book 
            (see documentation of page_raw_dict_reformat in get_char_df.py and char_df -> word_df 
            for more details)
        n: a positive ineger indicating the number of words in each word combination
        inplace: boolean value (defaults to True)
    OUTPUT:
        if inplace TRUE: 
            adds the result as columns named f'{n}_words' and f'{n}_flags' to df
        Returns a tuple of two series: (n_words_strings, n_flags)
    """
    #assumes n >= 1
    out_words_col = f"{n}_words"
    out_flags_col = f"{n}_flags"
    
    line_group_cols = ['vol_num', 'page_num', 
                       'block_num', 'block_num_absolute', 'block_bbox', 
                       'line_num', 'line_wmode', 'line_dir', 'line_bbox']
    
    n_words_lists = [None for i in range(n)]
    n_words_flags = [None for i in range(n)]

    n_words_lists[0] = df['word']
    n_words_flags[0] = df['span_flags']

    for i in range(1, n):
        n_words_lists[i] = df.groupby(line_group_cols)['word'].shift(-i, fill_value="")
        n_words_flags[i] = df.groupby(line_group_cols)['span_flags'].shift(-i, fill_value=0)

    zip_n_words = list(zip(*n_words_lists))
    n_word_string = list(map(lambda n_word_list : " ".join(n_word_list), zip_n_words))

    zip_n_flags = list(zip(*n_words_flags))
    combine_flags = list(map(lambda flag_list : reduce(lambda x, y: x | y, flag_list), zip_n_flags))
    
    if inplace == True:
        df[out_words_col] = n_word_string
        df[out_flags_col] = combine_flags
    return n_word_string, combine_flags


# NOTE: difflib_closest_match intended to have binomials in matching_lst
#       difflib_closest_match_score is also calculated AFTER difflib_closest_matches is used
#                                   meaning depending on how difflib.get_close_matches works
#                                   a potential match with a high difflib_closest_match_score
#                                   score won't be picked up.
def difflib_closest_matches(input_str, matching_lst):
    """ DOCUMENTATION
    given an input string and a list of strings to fuzzy match the input string with returns 
    the string closest match (according to difflib.get_close_matches function) and returns np.NaN 
    if no such matches exist. (here matching)
    INPUTS:
        input_str: String 
        matching_lst: List of Strings to match the input string with
    OUTPUT:
        if inplace TRUE: 
            adds the result as columns named f'{n}_words' and f'{n}_flags' to df
        Returns a tuple of two series: (n_words_strings, n_flags)
    """
    matched_list = difflib.get_close_matches(input_str, matching_lst)
    if len(matched_list) > 0:
        return matched_list[0]
    else:
        #print(input_str)
        return np.NaN


def difflib_closest_match_score(input_str, match_str):
    """ DOCUMENTATION
    given an 2 strings (input_str, match_str) finds the matching score (0 - 1) according to 
    after removing the spaces in the strings. 
    difflib.SequenceMatcher(None, input_str.lower(), match_str.lower()).ratio()
    INPUTS:
        input_str: String 
        match_str: String (possible np.NaN if a close match to input_str didn't exist
                           according to difflib_closest_matches)
    OUTPUT:
        Float between 0 to 1 (possible np.NaN if a close match to input_str didn't exist
                           according to difflib_closest_matches)
    """
    if isinstance(match_str, str):
        input_str = input_str.replace(' ', '')
        match_str = match_str.replace(' ', '')
        score = difflib.SequenceMatcher(None, input_str.lower(), match_str.lower()).ratio()
        return score
    else:
        return np.NaN
# ---------------------------------------------------------------- #

# ---------------------- adding the n_words ---------------------- #
i = 1
for vol_word_df, vol_species in [(vol1_word_df, vol_index_lists[0][1]), (vol2_word_df, vol_index_lists[1][1]), (vol3_word_df, vol_index_lists[2][1])]:
    print("VOLUME", str(i))
    for n in range(1, 5):
        word_group_col_name = f"{n}_words"
        match_col_name = f"{n}_words_match"
        match_score_col_name = f"{n}_words_match_score"
        print(f"finding best match for {word_group_col_name}")
        get_n_words_flagged(vol_word_df, n)
        vol_word_df[match_col_name] = vol_word_df[word_group_col_name].progress_apply(lambda x: difflib_closest_matches(x, vol_species))
        vol_word_df[match_score_col_name] = vol_word_df.progress_apply(lambda r: difflib_closest_match_score(r[word_group_col_name], r[match_col_name]), axis = 1)
    i += 1 
# ---------------------------------------------------------------- #

# ------------------------ finding entries ----------------------- #
print("finding boxes")
# NOT VERY ORGANIZED YET 
def get_section_id(row):
    if row['section_break'] == True:
        return row['line_id']
    else:
        return np.nan
def get_section_start_y(row):
    if row['section_break'] == True:
        return row['line_bbox'][1]
    else:
        return np.nan

likely_results = []
i = 1 
for vol_word_df, vol_genera in [(vol1_word_df, vol_index_lists[0][0]), (vol2_word_df, vol_index_lists[1][0]), (vol3_word_df, vol_index_lists[2][0])]:
    print("finding entry boxes for vol", str(i))
    vol_word_df['line_id'] = vol_word_df.progress_apply(lambda r : (r['page_num'], r['block_num'], r['line_num']), axis = 1) 
    is_binomial = ((~(vol_word_df['1_flags'].apply(is_italic)) & (vol_word_df['1_words_match_score'] > 0.85)) | 
                (~(vol_word_df['2_flags'].apply(is_italic)) & (vol_word_df['2_words_match_score'] > 0.85)) | 
                (~(vol_word_df['3_flags'].apply(is_italic)) & (vol_word_df['3_words_match_score'] > 0.85)) | 
                (~(vol_word_df['1_flags'].apply(is_italic)) & (vol_word_df['1_words_match_score'] > 0.85))) 
    vol_likely_results = vol_word_df[is_binomial]
    likely_results.append(vol_likely_results)

    is_stop = (((vol_word_df['word'].isin(vol_genera))) | 
            ((vol_word_df['line_bbox'].apply(lambda x : x[0] > 120)) & 
                (vol_word_df['word'].str.isupper()) & 
                (vol_word_df['pruned_word'].apply(len) > 2)))
    possible_stops = vol_word_df[is_stop]

    break_page_num =  vol_word_df[(is_binomial) | (is_stop)]['page_num']
    break_block_num = vol_word_df[(is_binomial) | (is_stop)]['block_num']
    break_line_num =  vol_word_df[(is_binomial) | (is_stop)]['line_num']
    break_id = list(zip(break_page_num, break_block_num, break_line_num))
    vol_word_df['section_break'] = vol_word_df['line_id'].isin(break_id)

    vol_word_df['section_id'] = vol_word_df.progress_apply(get_section_id, axis = 1)
    vol_word_df['section_id'].ffill(inplace=True)

    vol_word_df['section_start_y'] = vol_word_df.progress_apply(get_section_start_y, axis = 1)
    vol_word_df['section_start_y'].ffill(inplace=True)

    #getting section bbox 
    # break down line coords
    vol_word_df['line_x0'] = vol_word_df["line_bbox"].apply(lambda x: x[0])
    vol_word_df['line_y0'] = vol_word_df["line_bbox"].apply(lambda x: x[1])
    vol_word_df['line_x1'] = vol1_word_df["line_bbox"].apply(lambda x: x[2])
    vol_word_df['line_y1'] = vol_word_df["line_bbox"].apply(lambda x: x[3])

    #sections_coords: 
    vol_word_df["section_x0"] = vol_word_df.groupby('section_id')['line_x0'].transform('min')
    vol_word_df["section_y0_all"] = vol_word_df.groupby('section_id')['line_y0'].transform('min')
    vol_word_df["section_y0"] = vol_word_df[['section_y0_all','section_start_y']].max(axis=1)
    vol_word_df["section_x1"] = vol_word_df.groupby('section_id')['line_x1'].transform('max')
    vol_word_df["section_y1"] = vol_word_df.groupby('section_id')['line_y1'].transform('max')

    #section_bbox:
    vol_word_df["section_bbox"] = vol_word_df.apply(lambda r: (r["section_x0"], r["section_y0"], r["section_x1"], r["section_y1"]), axis = 1)

    #drop extra cols:
    vol_word_df.drop(columns= ["line_x0", "line_y0", "line_x1", "line_y1", "section_x0", "section_y0", "section_x1", "section_y1"], inplace = True)

    binom_page_num = vol_word_df[(is_binomial)]['page_num']
    binom_block_num = vol_word_df[(is_binomial)]['block_num']
    binom_line_num = vol_word_df[(is_binomial)]['line_num']
    binom_id = list(zip(binom_page_num, binom_block_num, binom_line_num))
    vol_word_df['binom_section'] = vol_word_df['line_id'].isin(binom_id)
    
    if not os.path.exists("../input/desc_box_df"):
        os.makedirs("../input/desc_box_df")

    vol_word_df.to_pickle(f"../input/desc_box_df/vol{i}_entry_df.pkl")
    i += 1
# ---------------------------------------------------------------- #
