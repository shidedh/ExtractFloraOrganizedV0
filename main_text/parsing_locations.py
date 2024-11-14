# %%
# October 2024 update

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cProfile import label #?not sure
import re
from fuzzywuzzy import fuzz
import difflib 
from fuzzywuzzy import process
import time
from tqdm import tqdm
import fitz

from functools import reduce
from fitz.utils import getColor

# %%
vol1_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 1.pdf'
vol2_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 2 COMPLETE.pdf'
vol3_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 3.pdf'

vol1_doc = fitz.open(vol1_path)
vol2_doc = fitz.open(vol2_path)
vol3_doc = fitz.open(vol3_path)

vol1_pages = [vol1_doc[i] for i in range(vol1_doc.page_count)]
vol2_pages = [vol2_doc[i] for i in range(vol2_doc.page_count)]
vol3_pages = [vol3_doc[i] for i in range(vol3_doc.page_count)]

# %%
vol1_char_df = pd.read_pickle("../input/char_df/vol1_df.pkl")
vol2_char_df = pd.read_pickle("../input/char_df/vol2_df.pkl")
vol3_char_df = pd.read_pickle("../input/char_df/vol3_df.pkl")

# %%
#mout. index: 
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

# %%
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

# %%
vol1_doc = fitz.open(vol1_path)
vol2_doc = fitz.open(vol2_path)
vol3_doc = fitz.open(vol3_path)
volume_doc = vol3_doc

# %%
#list of genera from index -- uppercased to match main text pattern

volume_index_df = vol3_index_df
volume_word_df = vol3_word_df
volume_char_df = vol3_char_df 
volume = "vol3"

vol_genera = volume_index_df[volume_index_df['taxon_rank'] == 'genus']['mouterde_genus'].str.upper().tolist()

#list of species binomial from main text
vol_species_temp_df = volume_index_df[(volume_index_df['taxon_rank'] == 'epithet') & (~volume_index_df['mouterde_genus'].isna())]
vol_species_binomial_list = list(zip(vol_species_temp_df['mouterde_genus'], vol_species_temp_df['mouterde_epithet']))
vol_species = list(map(lambda x: f"{x[0]} {x[1]}", vol_species_binomial_list))
vol_species_abriviation = list(map(lambda x: f"{x[0][0]}. {x[1]}", vol_species_binomial_list))

# %%
def is_italic(flags):
    return flags & 2 ** 1 != 0

# %%
def get_n_words_flagged(df, n, inplace = True):
    #assumes n > 1
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

# %%
tqdm.pandas()

# %%
volume_word_df = pd.read_pickle(f"../input/desc_box_df/{volume}_desc_df_v2.pkl")

# %%
is_binomial = ((~(volume_word_df['1_flags'].apply(is_italic)) & (volume_word_df['1_words_match_score'] > 0.85)) | 
               (~(volume_word_df['2_flags'].apply(is_italic)) & (volume_word_df['2_words_match_score'] > 0.85)) | 
               (~(volume_word_df['3_flags'].apply(is_italic)) & (volume_word_df['3_words_match_score'] > 0.85)) | 
               (~(volume_word_df['1_flags'].apply(is_italic)) & (volume_word_df['1_words_match_score'] > 0.85))) 
binom_page_num = volume_word_df[(is_binomial)]['page_num']
binom_block_num = volume_word_df[(is_binomial)]['block_num']
binom_line_num = volume_word_df[(is_binomial)]['line_num']
binom_id = list(zip(binom_page_num, binom_block_num, binom_line_num))

# %%
volume_char_df['line_id'] = volume_char_df.apply(lambda r : (r['page_num'], r['block_num'], r['line_num']), axis = 1)
volume_char_binom_df = volume_char_df[volume_char_df['line_id'].isin(binom_id)]

# %%
binom_char_width = volume_char_binom_df.groupby('line_id')['char_bbox'].transform(lambda x: x.apply(lambda y: y[2] - y[0])).mean()
binom_char_width

# %%
num_pages = volume_word_df['page_num'].max() + 1
for page_num in tqdm(range(num_pages)):
    volume_word_df.loc[volume_word_df['page_num'] == page_num, 'mean_binom_x0'] = volume_word_df[(volume_word_df['page_num'] == page_num) & (volume_word_df['line_id'].isin(binom_id))]['line_bbox'].apply(lambda x : x[0]).mean()

# %%
accepted_error = (binom_char_width)*2.5 #just eyeballing it ...

def is_binom_indentation(row):
    if abs(row['mean_binom_x0'] - (row['line_bbox'][0])) < accepted_error:
        return True
    else:
        return False

volume_word_df['is_binom_indentation'] = volume_word_df.progress_apply(is_binom_indentation, axis = 1)

# %%
volume_word_df['section_break']

# %%
volume_word_df['paragraph_id'] = np.nan
def paragraph_id(row):
    if row['is_binom_indentation'] == True:
        return row['line_id']
    if row['section_break'] == True:
        return row['line_id']
    else:
        return np.nan

volume_word_df['paragraph_id'] = volume_word_df.progress_apply(paragraph_id, axis = 1)
volume_word_df['paragraph_id'].ffill(inplace=True)

# %%
# break down line coords
volume_word_df['line_x0'] = volume_word_df["line_bbox"].apply(lambda x: x[0])
volume_word_df['line_y0'] = volume_word_df["line_bbox"].apply(lambda x: x[1])
volume_word_df['line_x1'] = volume_word_df["line_bbox"].apply(lambda x: x[2])
volume_word_df['line_y1'] = volume_word_df["line_bbox"].apply(lambda x: x[3])

#sections_coords: 
volume_word_df["paragraph_x0"] = volume_word_df.groupby(['page_num', 'paragraph_id'])['line_x0'].transform('min')
volume_word_df["paragraph_y0"] = volume_word_df.groupby(['page_num', 'paragraph_id'])['line_y0'].transform('min')
#volume_word_df["section_y0"] = vol1volume_word_df_word_df[['section_y0_all','section_y0_all']].max(axis=1)
volume_word_df["paragraph_x1"] = volume_word_df.groupby(['page_num', 'paragraph_id'])['line_x1'].transform('max')
volume_word_df["paragraph_y1"] = volume_word_df.groupby(['page_num', 'paragraph_id'])['line_y1'].transform('max')

#section_bbox:
volume_word_df["paragraph_bbox"] = volume_word_df.apply(lambda r: (r["paragraph_x0"], r["paragraph_y0"], r["paragraph_x1"], r["paragraph_y1"]), axis = 1)

#drop extra cols:
volume_word_df.drop(columns= ["line_x0", "line_y0", "line_x1", "line_y1", "paragraph_x0", "paragraph_y0", "paragraph_x1", "paragraph_y1"], inplace = True)

# %%
for page_num in tqdm(range(num_pages)):
    section_groups = volume_word_df[volume_word_df['page_num'] == page_num].groupby('section_id')
    page = volume_doc[page_num]
    colors = [getColor("plum"), getColor("orchid4")]
    
    paragraph_groups = volume_word_df[volume_word_df['page_num'] == page_num].groupby('paragraph_id')
    for name, paragraph in paragraph_groups:
        i = 0
        paragraph_id = paragraph.iloc[0]['paragraph_id']
        paragraph_section_id = paragraph.iloc[0]['section_id']
        paragraph_bbox = paragraph.iloc[0]['paragraph_bbox']
        is_L_loc = paragraph.iloc[0]['word'].lower() in ["l.", "l"]
        is_S_loc = paragraph.iloc[0]['word'].lower() in ["s.", "s"]
        
        c = getColor("lightgray")
        if is_L_loc: 
            c = getColor("lightblue")
        if is_S_loc:
            c = getColor("pink")
        if paragraph_section_id in binom_id:
            r_box = fitz.Rect(paragraph_bbox)
            annot_rect = page.add_rect_annot(r_box)
            annot_rect.set_colors({"stroke":c})
            annot_rect.update()
            i += 1
        c = getColor("lightgray")

    for name, section in section_groups:
        section_id = section.iloc[0]['section_id']
        section_bbox = section.iloc[0]['section_bbox']
        if section_id in binom_id:
            r_box = fitz.Rect(section_bbox)
            #r_box.set_stroke_color(stroke=getColor("violetred4"))
            annot_rect = page.add_rect_annot(r_box)
            annot_rect.set_colors({"stroke": getColor("violetred4")})
            annot_rect.update()

marked_epithet_fname = f"../output/Oct2024/{volume}_binom_sections_paragraphs_LS_v1.pdf"
volume_doc.save(marked_epithet_fname)

# %%
volume_word_df.columns

# %%
volume_word_df['pruned_word']

# %%
[getColor("darkpink"), getColor("lightblue")]

# %%
from PIL import Image

# Create a new image with a solid color
width = 50
height = 50
colors = [getColor("pink"), getColor("lightblue"),  getColor("lightgray")]
for color_1 in colors:
    color_255 = tuple(int(c_val*255) for c_val in color_1) # red color
    image = Image.new("RGB", (width, height), color_255)
    image.show()
# Display the image
image

# %% [markdown]
# ### parse L. and S. 

# %%
paragraph_groups = volume_word_df.groupby('paragraph_id')
volume_word_df['paragraph_word_num'] = paragraph_groups.cumcount() + 1
paragraph_groups_123 = volume_word_df[volume_word_df['page_num'] == 123].groupby('paragraph_id')

for name, paragraph in paragraph_groups_123:
    is_L_loc = paragraph.iloc[0]['word'].lower() in ["l.", "l"]
    if is_L_loc:
        break

# %%
# VOL1 ONLY
# checking if paragraph_word_num works correctly when section paragraph acrros 2 pages accross
# HC_id = volume_word_df[volume_word_df['4_words'] == "Heteropogon con tort us"]['paragraph_id'].iloc[0]
# volume_word_df[(volume_word_df['paragraph_id'] == HC_id)]['paragraph_word_num']

# %%
" ".join(paragraph['word'])

# %%
paragraph

# %%
paragraph_italics_list = (paragraph[paragraph['span_flags'].apply(is_italic)]['paragraph_word_num'] -1).tolist()
paragraph_italics_list.append((paragraph['paragraph_word_num']).max())

sub_loc_dict = {}
paragraph_text = paragraph['word'].tolist()

for i in range(len(paragraph_italics_list) - 1):
    curr_word_i = int(paragraph_italics_list[i])
    next_word_i = int(paragraph_italics_list[i+1]) #only works if the last word of L. is not italics
    sub_loc_italics = paragraph_text[curr_word_i]
    sub_location_list = paragraph_text[curr_word_i + 1: next_word_i]
    sub_location_str = " ".join(sub_location_list)
    # make dict for L. / S. whenever it exists.
    sub_loc_result = [match_s.strip() for match_s in re.findall(r'([^,()]+?)(?=[,(])(?![^()]*\))', sub_location_str)]
    paragraph_section_id = paragraph['section_id'].iloc[0]
    try:
        sub_loc_dict[paragraph_section_id]
    except:
        sub_loc_dict[paragraph_section_id] = {}
    if paragraph_text[0].lower() in ['l.', 'l']:
        try: 
            sub_loc_dict[paragraph_section_id]['L.']
        except: 
            sub_loc_dict[paragraph_section_id]['L.'] = []
        sub_loc_dict[paragraph_section_id]['L.'].append({sub_loc_italics: sub_loc_result})
    if paragraph_text[0].lower() in ['s.', 's']:
        try: 
            sub_loc_dict[paragraph_section_id]['S.']
        except: 
            sub_loc_dict[paragraph_section_id]['S.'] = []
        sub_loc_dict[paragraph_section_id]['S.'].append({sub_loc_italics: sub_loc_result})

# %%
marked_epithet_fname = f"../output/Oct2024/{volume}_binom_sections_paragraphs_LS_parsed_v0.pdf"
volume_doc.save(marked_epithet_fname)

# %%


# %%
from unidecode import unidecode

# %%
def difflib_closest_match_alnum_unidecode_score(input_str, match_str):
    if isinstance(match_str, str):
        input_str = unidecode("".join([c for c in input_str if c.isalnum()]))
        match_str = unidecode("".join([c for c in match_str if c.isalnum()]))
        
        score = difflib.SequenceMatcher(None, input_str.lower(), match_str.lower()).ratio()
        return score
    else:
        return np.NaN

# %%
paragraph_groups = volume_word_df.groupby('paragraph_id')
volume_word_df['paragraph_word_num'] = paragraph_groups.cumcount() + 1
paragraph_groups_123 = volume_word_df[volume_word_df['page_num'] == 123].groupby('paragraph_id')

sub_loc_dict = {}
other_data = {}
for name, paragraph in tqdm(paragraph_groups):
    paragraph_italics_list = (paragraph[paragraph['span_flags'].apply(is_italic)]['paragraph_word_num'] -1).tolist()
    paragraph_italics_list.append((paragraph['paragraph_word_num']).max())

    paragraph_text = paragraph['word'].tolist()
    no_italics_line = False
    if paragraph_italics_list[0] != 1:
        paragraph_italics_list = [0] + paragraph_italics_list
        no_italics_line = True
        
    for i in range(len(paragraph_italics_list) - 1):
        curr_word_i = int(paragraph_italics_list[i])
        next_word_i = int(paragraph_italics_list[i+1]) #only works if the last word of L. is not italics
        sub_loc_italics = paragraph_text[curr_word_i]
        sub_location_list = paragraph_text[curr_word_i + 1: next_word_i]
        sub_location_str = " ".join(sub_location_list)
        # make dict for L. / S. whenever it exists.
        
        sub_loc_result = [match_s.strip() for match_s in re.findall(r'([^,()]+?)(?=[,(])(?![^()]*\))', sub_location_str)]

        if no_italics_line:
            sub_loc_italics = 'NO ITALICS'
            sub_location_list = paragraph_text[curr_word_i: next_word_i]
            sub_loc_result = [" ".join(sub_location_list)]
                
        paragraph_section_id = paragraph['section_id'].iloc[0]
        try:
            sub_loc_dict[paragraph_section_id]
        except:
            sub_loc_dict[paragraph_section_id] = {}
        
        try:
            other_data[paragraph_section_id]
        except:
            other_data[paragraph_section_id] = {}
        
        aire_geogr_match = difflib_closest_match_alnum_unidecode_score(" ".join(paragraph_text[:2]), "aire géogr.") > 0.9
        L_match = paragraph_text[0].lower() in ['l.', 'l']
        S_match = paragraph_text[0].lower() in ['s.', 's']
        floraison_match = difflib_closest_match_alnum_unidecode_score(paragraph_text[0], "Floraison:") > 0.9
        is_description = paragraph_section_id == name
        if L_match:
            try: 
                sub_loc_dict[paragraph_section_id]['L.']
            except: 
                sub_loc_dict[paragraph_section_id]['L.'] = {}
            
            try:
                sub_loc_dict[paragraph_section_id]['L.'][sub_loc_italics]
            except:
                sub_loc_dict[paragraph_section_id]['L.'][sub_loc_italics] = []
            sub_loc_dict[paragraph_section_id]['L.'][sub_loc_italics].extend(sub_loc_result)
        
        elif S_match:
            try: 
                sub_loc_dict[paragraph_section_id]['S.']
            except: 
                sub_loc_dict[paragraph_section_id]['S.'] = {}

            try:
                sub_loc_dict[paragraph_section_id]['S.'][sub_loc_italics]
            except:
                sub_loc_dict[paragraph_section_id]['S.'][sub_loc_italics] = []
            sub_loc_dict[paragraph_section_id]['S.'][sub_loc_italics].extend(sub_loc_result)
        
        elif aire_geogr_match:
            other_data[paragraph_section_id]["aire géogr."] = " ".join(paragraph_text[2:])
        
        elif floraison_match:
            other_data[paragraph_section_id]["Floraison"] = " ".join(paragraph_text[1:])
        
        elif is_description:
            other_data[paragraph_section_id]["desc_paragraph"] = " ".join(paragraph_text)
        else:
            try:
                other_data[paragraph_section_id]["other"]
            except:
                other_data[paragraph_section_id]["other"] = [] 
            other_data[paragraph_section_id]["other"].append([" ".join(paragraph_text)])

# %%
is_binomial = ((~(volume_word_df['1_flags'].apply(is_italic)) & (volume_word_df['1_words_match_score'] > 0.85)) | 
               (~(volume_word_df['2_flags'].apply(is_italic)) & (volume_word_df['2_words_match_score'] > 0.85)) | 
               (~(volume_word_df['3_flags'].apply(is_italic)) & (volume_word_df['3_words_match_score'] > 0.85)) | 
               (~(volume_word_df['1_flags'].apply(is_italic)) & (volume_word_df['1_words_match_score'] > 0.85))) 

def get_binomial_string(row):
    all_combos = []
    for i in range(1, 5):
        all_combos.append((row[f'{i}_flags'], row[f'{i}_words'], row[f'{i}_words_match_score']))
    non_italics_combs = [comb for comb in all_combos if is_italic(comb[0])]
    if non_italics_combs:
        binom_name = max(non_italics_combs, key = lambda x : x[2])[1] #not using closest match because that sometimes isn't right
        return binom_name 
    else: 
        return np.nan

#vol1_word_df.apply(get_binomial_string, axis = 1).groupby('section_id').transform('max')
volume_word_df['binom_string'] = volume_word_df.apply(get_binomial_string, axis = 1)

# %%
is_binomial = ((~(volume_word_df['1_flags'].apply(is_italic)) & (volume_word_df['1_words_match_score'] > 0.85)) | 
               (~(volume_word_df['2_flags'].apply(is_italic)) & (volume_word_df['2_words_match_score'] > 0.85)) | 
               (~(volume_word_df['3_flags'].apply(is_italic)) & (volume_word_df['3_words_match_score'] > 0.85)) | 
               (~(volume_word_df['1_flags'].apply(is_italic)) & (volume_word_df['1_words_match_score'] > 0.85))) 
volume_word_df[is_binomial]

# %%
volume_word_df[(is_binomial==True) & (volume_word_df['page_num'] >= 78) & (volume_word_df['page_num'] <= 606)]

# %%
len(volume_word_df[(is_binomial==True) & (volume_word_df['page_num'] >= 78) & (volume_word_df['page_num'] <= 606)].groupby('section_id'))

# %%
volume_word_df[(volume_word_df['binom_section']) &  (volume_word_df['page_num'] >= 78) & (volume_word_df['page_num'] <= 606)].groupby('section_id').first()['binom_string']

# %%
volume_word_df.columns

# %%
def get_binomial_string(row):
    all_combos = []
    for i in range(1, 5):
        all_combos.append((row[f'{i}_flags'], row[f'{i}_words'], row[f'{i}_words_match_score']))
    non_italics_combs = [comb for comb in all_combos if is_italic(comb[0])]
    if non_italics_combs:
        binom_name = max(non_italics_combs, key = lambda x : x[2])[1] #not using closest match because that sometimes isn't right
        return binom_name 
    else: 
        return np.nan

# %%
section_groups = volume_word_df.groupby('section_id')

# ### get binomial name the way I did it in Aaron's code :) 

# %%
def get_binomial(row):
    combs = [(row['1_words'], row['1_flags'], row['1_words_match_score']),
             (row['2_words'], row['2_flags'], row['2_words_match_score']),
             (row['3_words'], row['3_flags'], row['3_words_match_score']),
             (row['4_words'], row['4_flags'], row['4_words_match_score'])]
    combs_valid = [c for c in combs if (is_italic(c[1]) == False and np.isnan(c[2]) == False)]
    return max(combs_valid, key = lambda x: x[2])[0]

# %%
section_groups = volume_word_df[volume_word_df['page_num']<=609].groupby('section_id')
fake_span = ''
fake_line = ''
fake_block = ''
fake_warning = ''
items = []
boxes = []

volume_word_df['binom_bbox'] = np.nan

for name, section in tqdm(section_groups):
    page_num = int(name[0])
    section_id = section.iloc[0]['section_id']
    if section_id in binom_id:
        section_bbox = section.iloc[0]['section_bbox']
        desc_rect = section_bbox
        binom_section = section[((~(section['1_flags'].apply(is_italic)) & (section['1_words_match_score'] > 0.85)) | 
                                 (~(section['2_flags'].apply(is_italic)) & (section['2_words_match_score'] > 0.85)) | 
                                 (~(section['3_flags'].apply(is_italic)) & (section['3_words_match_score'] > 0.85)) | 
                                 (~(section['1_flags'].apply(is_italic)) & (section['1_words_match_score'] > 0.85)))]
        binom = binom_section.apply(get_binomial, axis = 1).iloc[0]
        binom = "".join(binom)
        num_binom_words = len(binom.split(' '))
        binom_index = []
        #probably not best way of doing this 
        start_binom_index = binom_section.index[0]
        start_binom_i = list(section.index).index(start_binom_index)
        for i in range(num_binom_words):
            section.index
            binom_index.append(section.index[start_binom_i])
        binom_x0 = section.loc[binom_index, 'word_bbox'].apply(lambda x : x[0]).min()
        binom_y0 = section.loc[binom_index, 'word_bbox'].apply(lambda x : x[1]).min()
        binom_x1 = section.loc[binom_index, 'word_bbox'].apply(lambda x : x[2]).max()
        binom_y1 = section.loc[binom_index, 'word_bbox'].apply(lambda x : x[3]).max()
        binom_rect = fitz.Rect((binom_x0, binom_y0, binom_x1, binom_y1))
        
        
        span_num = binom_section.iloc[0]['span_num']
        line_num = binom_section.iloc[0]['line_num']
        block_num = binom_section.iloc[0]['block_num']

        section_shape = volume_word_df.loc[volume_word_df['section_id'] == section_id].shape[0]
        volume_word_df.loc[volume_word_df['section_id'] == section_id, 'binom_name'] = binom
        volume_word_df.loc[(volume_word_df['section_id'] == section_id), 'binom_bbox'] = volume_word_df.loc[(volume_word_df['section_id'] == section_id)].apply(lambda _ : (binom_x0, binom_y0, binom_x1, binom_y1), axis = 1)
        # this line was not working the normal way ... seems fine now tho so yay


# %%
dict_test = {'L.': {'Ctlitt.': ['Plage de Khaldé'],
        'Ct.':     ['Beyrouth', 'adventice en pleine ville']},
 'S.': {'Haur.':   ["Ezra'a", 'Sanamein'], 
        'J.D.': ['Mourdouk']}}
# turning this into a csv thing is left

{
 'country':      [country for country in dict_test.keys() for general_loc in dict_test[country] for specific_loc in dict_test[country][general_loc]],
 'general_loc':  [general_loc for country in dict_test.keys() for general_loc in dict_test[country] for specific_loc in dict_test[country][general_loc]],
 'specific_loc': [specific_loc for country in dict_test.keys() for general_loc in dict_test[country] for specific_loc in dict_test[country][general_loc]]
}

sub_loc_dict

# %%
k = 'L.'
len(dict_test[k].values())


for country in dict_test.keys():
    for general_loc in dict_test[country]:
        for specific_loc in dict_test[country][general_loc]:
            print(country)

[country for country in dict_test.keys() for general_loc in dict_test[country] for specific_loc in dict_test[country][general_loc]]

# %%
for country in dict_test.keys():
    for general_loc in dict_test[country]:
        for specific_loc in dict_test[country][general_loc]:
            print(general_loc)

# %%


# %%
all_sections_df = []
for section_id in tqdm(sub_loc_dict):
    dict_test = sub_loc_dict[section_id]
    dict_else = other_data[section_id]

    num_loc_data = len([country for country in dict_test.keys() for general_loc in dict_test[country] for specific_loc in dict_test[country][general_loc]])
    # if no location data not counted? 
    name = volume_word_df[(volume_word_df['section_id'] == section_id)]['binom_name'].iloc[0]
    name_data = [name] * num_loc_data
    page_num_data = [section_id[0]] * num_loc_data
    country_data = [country for country in dict_test.keys() for general_loc in dict_test[country] for specific_loc in dict_test[country][general_loc]]

    aire_geor = [np.NaN] * num_loc_data
    floraison = [np.NaN] * num_loc_data
    desc = [np.NaN] * num_loc_data
    other = [np.NaN] * num_loc_data
    basic_author = [np.NaN] * num_loc_data
    author = np.NaN
    for k in dict_else:
        if k == "aire géogr.":
            aire_geor = [dict_else["aire géogr."]] * num_loc_data
        elif k == "Floraison":
            floraison = [dict_else["Floraison"]] * num_loc_data
        elif k == "desc_paragraph":
            desc = [dict_else["desc_paragraph"]] * num_loc_data
            if isinstance(name, str):
                author = " ".join(dict_else["desc_paragraph"].split("—")[0].split(" ")[len(name.split(" ")):])

                text = author
                if len(re.findall(r"\(.*?\)", text)) > 0:
                    #isn't perfect because of infra species info on fisrt line + its authors
                    first_paran = re.findall(r"\(.*?\)", text)[0]
                    first, rest = text.split(first_paran, 1)
                    rest = re.sub(r"\(.*?\)*", "", rest)
                    if rest == '':
                        author = first + first_paran
                    if first == '':
                        author = first_paran + rest
                    else:
                        author = first
            
            basic_author = [author]* num_loc_data
        elif k == "other":
            other = [dict_else["other"]] * num_loc_data

    if isinstance(name, str):
        section_data = {'page_num': page_num_data,
                        'binomial': name_data,
                        'basic_author': basic_author,
                        'desc_paragraph': desc,
                        'floraison': floraison,
                        'aire_geor': aire_geor,
                        'other': other,
                        'country':      [country for country in dict_test.keys() for general_loc in dict_test[country] for specific_loc in dict_test[country][general_loc]],
                        'general_loc':  [general_loc for country in dict_test.keys() for general_loc in dict_test[country] for specific_loc in dict_test[country][general_loc]],
                        'specific_loc': [specific_loc for country in dict_test.keys() for general_loc in dict_test[country] for specific_loc in dict_test[country][general_loc]]
                    }
        df = pd.DataFrame.from_dict(section_data)
        all_sections_df.append(df)

# %%
result_df = pd.concat(all_sections_df)

# %%
result_df['basic_author'].unique()

# %%
test = result_df[result_df['binomial'] == 'Isoetes olympica']
name =  'Isoetes olympica'

# %%
result_df[result_df['page_num'] == 79]

# %%
result_df.to_csv(f'../output/Oct2024/{volume}_location_parsed_v4.csv')
