# %%
import fitz
import numpy as np
import pandas as pd
from tqdm import tqdm

import io
from PIL import Image, ImageDraw, ImageFont, ImageColor

import math
import re

# %% [markdown]
# ### IMPORTING BOOKS

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

vol1_index = list(range(616, 639)) #inclusive
vol2_index = list(range(703, 725 + 8))
vol3_index = list(range(555, 583))

# %% [markdown]
# #### Setting Global parameters

# %%
TARGET_DPI = 300
mat = fitz.Matrix(TARGET_DPI/ 72, TARGET_DPI/ 72)

# %% [markdown]
# ### Finding strict matching genera, epithet, and column numbers

# %%
def genus_match(row):
    word_rspace_removed = row['word']
    return row['word_num'] == 0 and \
           word_rspace_removed.isalpha() and \
           word_rspace_removed[0].isupper() and word_rspace_removed[1:].islower()
           
def epithet_match(row):
    word_rspace_removed = row['word']
    return row['word_num'] == 0 and \
           word_rspace_removed.isalpha() and \
           word_rspace_removed.islower()

# %%
#rightmost point of any bounding box:
def get_center_x0(vol_char_df, page_num, bias = 30):
    """WARNING: Bias = 30 large bias causes miscatagorization in page number in book"""
    df = vol_char_df[vol_char_df['page_num'] == page_num]
    
    right_bound = df['line_bbox'].apply(lambda x : x[2]).max() 
    #leftmost point of any bounding box:
    left_bound = df['line_bbox'].apply(lambda x : x[0]).min()

    return 0.5*(right_bound + left_bound) - bias


def get_col_num(coords, center_x0):
    x0, y0, x1, y1 = coords
    return int(x0 >= center_x0)

# %%
all_vol_data_col_num = [(vol1_char_df, vol1_index, vol1_doc),
                        (vol2_char_df, vol2_index, vol2_doc),
                        (vol3_char_df, vol3_index, vol3_doc)]

for vol_char_df ,vol_index, doc in all_vol_data_col_num: 
    #for each volume check if genus pattern / epithet pattern exists within the index part of the book
    vol_char_df['genus_index_pat_match'] = (vol_char_df['page_num'].isin(vol_index)) & (vol_char_df.apply(genus_match, axis = 1))
    vol_char_df['epithet_index_pat_match'] = (vol_char_df['page_num'].isin(vol_index)) & (vol_char_df.apply(epithet_match, axis = 1))
    
    for page_num in tqdm(vol_index):
        center_x0 = get_center_x0(vol_char_df, page_num)
        #find center based on x0 coordinate of each line
        vol_char_df['col_num'] = vol_char_df['line_bbox'].apply(lambda coords : get_col_num(coords, center_x0)) 

# %% [markdown]
# ### Genus / epithet flagging 
# flagging pages where number of strict genus or epithet patern matches is less than 3 per column

# %%
all_vol_data_flagg_strict_match = [(vol1_char_df, vol1_index, vol1_doc, "strickt_match_vol1"),
                                   (vol2_char_df, vol2_index, vol2_doc, "strickt_match_vol2"),
                                   (vol3_char_df, vol3_index, vol3_doc, "strickt_match_vol3")]

for vol_char_df, vol_index, vol_doc, output_name in all_vol_data_flagg_strict_match: 
    #for each volume 
    image_list = []
    genus_flag_list = []
    epithet_flag_list = []
    for page_num in tqdm(vol_index):
        pix_map = vol_doc.get_page_pixmap(page_num,matrix=mat)
        image = Image.open(io.BytesIO(pix_map.tobytes()))
        draw = ImageDraw.Draw(image)

        genus_db = vol_char_df[(vol_char_df['page_num'] == page_num)
                                & (vol_char_df['genus_index_pat_match'] == True)
                            ].loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin",	"char_bbox"])
                            ].drop_duplicates()

        epithet_db = vol_char_df[(vol_char_df['page_num'] == page_num) 
                                & (vol_char_df['epithet_index_pat_match'] == True)
                                ].loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin",	"char_bbox"])
                                ].drop_duplicates()

        #genus pattern match flag should check with half page and not entire page:
        for col in range(2):
            num_genus_col = genus_db[genus_db["col_num"] == col].shape[0]
            num_epithet_col = epithet_db[epithet_db["col_num"] == col].shape[0]
            if num_genus_col <= 2:
                genus_flag_list.append((num_genus_col, page_num - vol_index[0] + 1, col))
            if num_epithet_col <= 2:
                epithet_flag_list.append((num_epithet_col, page_num - vol_index[0] + 1, col))

        for coord in genus_db['word_bbox']:
            x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
            draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#FF7F50"), width=5)

        for coord in epithet_db['word_bbox']:
            x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
            draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#003399"), width=5)

        image_list.append(image)

    image_list[0].save('../output/local/'+output_name+'.pdf' ,save_all=True, append_images=image_list[1:])    
    
    num_flag_pages = len(set([g[1] for g in genus_flag_list] + [e[1] for e in epithet_flag_list]))
    if num_flag_pages > 0: 
        print("***FLAGS***")
        print(f" number of pages to check: {num_flag_pages}")
        if genus_flag_list:
            print("  genera")
            [print(f"\t number of genera: {g_flag[0]}, page number: {g_flag[1]}, column number: {g_flag[2]}") for g_flag in genus_flag_list]
        if epithet_flag_list:
            print("  epithets")
            [print(f"\t number of epithets: {e_flag[0]}, page number: {e_flag[1]}, column number: {e_flag[2]}") for e_flag in epithet_flag_list]

# %% [markdown]
# Based on flags need to make sure: 
# - first find epithet coord match 
# - then find genus coord match s.t. word is not in epithet coord match

# %% [markdown]
# ### match based on coordinates

# %%
def is_coord_match(x, x_ref_left, x_ref_right, margin):
    return (x_ref_left - margin <= x[0] and x[0] <= x_ref_left + margin) or (x_ref_right - margin <= x[0] and x[0] <= x_ref_right + margin)

# %% [markdown]
# #### epithets

# %%
all_vol_data_coord_match = [(vol1_char_df, vol1_index),
                            (vol2_char_df, vol2_index),
                            (vol3_char_df, vol3_index)]

for vol_char_df, vol_index in all_vol_data_coord_match: 
    vol_char_df["epithet_coord_match"] = vol_char_df["word_bbox"].apply(lambda x : False)
    for page_num in tqdm(vol_index):
        margin = 1.25 * vol_char_df[(vol_char_df["epithet_index_pat_match"] == True)]["char_bbox"].apply(lambda x : x[2] - x[0]).mean()
        epithet_char_df = vol_char_df[(vol_char_df["page_num"] == page_num) & (vol_char_df["epithet_index_pat_match"] == True)]
        epithet_df = epithet_char_df.loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin", "char_bbox"])].drop_duplicates()
        page_epithet_2dic = [{}, {}]
        
        for i in range(epithet_df.shape[0]):
            e_index = str(page_num) + "_" + str(i)
            p0 = epithet_df['word_bbox'].iloc[i]
            x_ref = p0[0]
            col = epithet_df['col_num'].iloc[i]

            ref_neighbors_df = epithet_df[(epithet_df["page_num"] == page_num) & 
                                          (epithet_df["word_bbox"].apply(lambda x : x_ref - margin <= x[0] and x[0] <= x_ref + margin))]
            
            num_neighbors = ref_neighbors_df.shape[0]
            mean_neighbors = ref_neighbors_df["word_bbox"].apply(lambda x : x[0]).mean()
            page_epithet_2dic[col][e_index] = (num_neighbors, mean_neighbors)
        
        mean_left_epithet = max(page_epithet_2dic[0].values(), default = [-1, -1])[1]
        mean_right_epithet = max(page_epithet_2dic[1].values(), default = [-1, -1])[1]

        if mean_left_epithet == -1 or mean_right_epithet == -1:
            mean_valid_col = max(mean_left_epithet, mean_right_epithet)
            vol_char_df.loc[(vol_char_df["page_num"] == page_num) , "epithet_coord_match"] = vol_char_df[(vol_char_df["page_num"] == page_num)]["pruned_word_bbox"].apply(lambda x : is_coord_match(x, mean_valid_col, mean_valid_col, margin))
        elif mean_left_epithet == -1 and mean_right_epithet == -1:
            vol_char_df.loc[(vol_char_df["page_num"] == page_num) , "epithet_coord_match"] = vol_char_df[(vol_char_df["page_num"] == page_num)]["pruned_word_bbox"].apply(lambda x : False)
        else: 
            vol_char_df.loc[(vol_char_df["page_num"] == page_num) , "epithet_coord_match"] = vol_char_df[(vol_char_df["page_num"] == page_num)]["pruned_word_bbox"].apply(lambda x : is_coord_match(x, mean_left_epithet, mean_right_epithet, margin))

# %%
all_vol_data_epithet_coord_match_test = [(vol1_char_df, vol1_index, vol1_doc, "epithet_coord_match_pruned_vol1"),
                                         (vol2_char_df, vol2_index, vol2_doc, "epithet_coord_match_pruned_vol2"),
                                         (vol3_char_df, vol3_index, vol3_doc, "epithet_coord_match_pruned_vol3")]

for vol_char_df, vol_index, doc, output_name in all_vol_data_epithet_coord_match_test: 
    #for each volume 
    image_list = []
    
    for page_num in tqdm(vol_index):
        pix_map = doc.get_page_pixmap(page_num,matrix=mat)
        image = Image.open(io.BytesIO(pix_map.tobytes()))
        draw = ImageDraw.Draw(image)
        
        epithet_coord_db = vol_char_df[(vol_char_df['page_num'] == page_num) & 
                                     (vol_char_df['epithet_coord_match'] == True)
                            ].loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin",	"char_bbox"])
                            ].drop_duplicates()

        epithet_db = vol_char_df[(vol_char_df['page_num'] == page_num) 
                                & (vol_char_df['epithet_index_pat_match'] == True)
                                ].loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin",	"char_bbox"])
                                ].drop_duplicates()

        #epithet Coord is orange-pinkish, 5
        for coord in epithet_coord_db["pruned_word_bbox"] :
            x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
            draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#FF7F50"), width=5)

        #epithet is blue, 3
        for coord in epithet_db['word_bbox'] :
            x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
            draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#003399"), width=3)
        image_list.append(image)

    #save pages of the volume
    image_list[0].save('../output/local/'+output_name+'.pdf' ,save_all=True, append_images=image_list[1:])

# %%
# Reminder:
# all_vol_data_coord_match = [(vol1_char_df, vol1_index),
#                             (vol2_char_df, vol2_index),
#                             (vol3_char_df, vol3_index)]
# DOES NOT CHECK IF COORD IS SAME AS EPITHET UNTIL NEXT SECTION!

for vol_char_df, vol_index in all_vol_data_coord_match: 
    #genus and not epithet
    vol_char_df["genus_coord_match"] = vol_char_df["word_bbox"].apply(lambda x : False)
    for page_num in tqdm(vol_index):
        margin = 1.25 * vol_char_df[(vol_char_df["genus_index_pat_match"] == True)]["char_bbox"].apply(lambda x : x[2] - x[0]).mean()
        genus_char_df = vol_char_df[(vol_char_df["page_num"] == page_num) &
                                    (vol_char_df["genus_index_pat_match"] == True)]
        genus_df = genus_char_df.loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin", "char_bbox"])].drop_duplicates()
        page_genus_2dic = [{}, {}]
        
        epithet_left_coord_mean = vol_char_df[(vol_char_df["epithet_coord_match"] == True) &
                                              (vol_char_df["page_num"] == page_num) &
                                              (vol_char_df["col_num"] == 0)
                                             ]['pruned_word_bbox'].apply(lambda x : x[0]).mean()
        epithet_right_coord_mean = vol_char_df[(vol_char_df["epithet_coord_match"] == True) &
                                               (vol_char_df["page_num"] == page_num) &
                                               (vol_char_df["col_num"] == 1)
                                             ]['pruned_word_bbox'].apply(lambda x : x[0]).mean()
        epithet_coord_mean_list = [epithet_left_coord_mean, epithet_right_coord_mean]

        for i in range(genus_df.shape[0]):
            g_index = str(page_num) + "_" + str(i)
            p0 = genus_df['word_bbox'].iloc[i]
            x_ref = p0[0]
            col = genus_df['col_num'].iloc[i]

            ref_neighbors_df = genus_df[(genus_df["page_num"] == page_num) & 
                                        (genus_df["word_bbox"].apply(lambda x : x_ref - margin <= x[0] and x[0] <= x_ref + margin))]

            num_neighbors = ref_neighbors_df.shape[0]
            mean_neighbors = ref_neighbors_df["word_bbox"].apply(lambda x : x[0]).mean()
            if mean_neighbors > epithet_coord_mean_list[col]: 
                mean_neighbors = -1
            page_genus_2dic[col][g_index] = (num_neighbors, mean_neighbors)
        
        mean_left_genus = max(page_genus_2dic[0].values(), default = [-1, -1])[1]
        mean_right_genus = max(page_genus_2dic[1].values(), default = [-1, -1])[1]

        if mean_left_genus == -1 or mean_right_genus == -1:
            mean_valid_col = max(mean_left_genus, mean_right_genus)
            vol_char_df.loc[(vol_char_df["page_num"] == page_num) , "genus_coord_match"] = vol_char_df[(vol_char_df["page_num"] == page_num)]["pruned_word_bbox"].apply(lambda x : is_coord_match(x, mean_valid_col, mean_valid_col, margin))
        elif mean_left_genus == -1 and mean_right_genus == -1:
            vol_char_df.loc[(vol_char_df["page_num"] == page_num) , "genus_coord_match"] = vol_char_df[(vol_char_df["page_num"] == page_num)]["pruned_word_bbox"].apply(lambda x : False)
        else: 
            vol_char_df.loc[(vol_char_df["page_num"] == page_num) , "genus_coord_match"] = vol_char_df[(vol_char_df["page_num"] == page_num)]["pruned_word_bbox"].apply(lambda x : is_coord_match(x, mean_left_genus, mean_right_genus, margin))

# %%
all_vol_data_genus_coord_match_test = [(vol1_char_df, vol1_index, vol1_doc, "genus_coord_match_vol1"),
                                       (vol2_char_df, vol2_index, vol2_doc, "genus_coord_match_vol2"),
                                       (vol3_char_df, vol3_index, vol3_doc, "genus_coord_match_vol3")]

for vol_char_df, vol_index, doc, output_name in all_vol_data_genus_coord_match_test: 
    #for each volume 
    image_list = []

    for page_num in tqdm(vol_index):
        pix_map = doc.get_page_pixmap(page_num,matrix=mat)
        image = Image.open(io.BytesIO(pix_map.tobytes()))
        draw = ImageDraw.Draw(image)
        

        genus_coord_db = vol_char_df[(vol_char_df['page_num'] == page_num) & 
                                     (vol_char_df['genus_coord_match'] == True)
                            ].loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin",	"char_bbox"])
                            ].drop_duplicates()

        epithet_db = vol_char_df[(vol_char_df['page_num'] == page_num) 
                                & (vol_char_df['epithet_coord_match'] == True)
                                ].loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin",	"char_bbox"])
                                ].drop_duplicates()

        #genus Coord is orange-pinkish, 5
        for coord in genus_coord_db['word_bbox'] :
            x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
            draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#FF7F50"), width=5)
            
        # #epithet is red, 3
        for coord in epithet_db['word_bbox'] :
            x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
            draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#000099"), width=3)
        image_list.append(image)

    #save pages of the volume
    image_list[0].save('../output/local/'+output_name+'.pdf' ,save_all=True, append_images=image_list[1:])

# %% [markdown]
# #### improving the coord matches 
# takes genus coming before epithet into account now

# %%
def potential_genus_match(row):
    word_rspace_removed = row['word']
    return row['genus_coord_match'] == True and \
           row['epithet_coord_match'] == False and \
           word_rspace_removed.find("Flore") == -1 and \
           ((word_rspace_removed.isupper() == False and \
             word_rspace_removed.isnumeric() == False) or \
            ((word_rspace_removed == 'X') or (word_rspace_removed =='×')))
           # removing this for-    hg now ... and row['genus_mean_coord'] < row['epithet_mean_coord'] #important to check this only when epithet_coord_match is false?

def potential_epithet_match(row):
    word_rspace_removed = row['word']
    return row['epithet_coord_match'] == True and \
           ((word_rspace_removed.isupper() == False and \
             word_rspace_removed.isnumeric() == False) or \
            (word_rspace_removed == 'X') or (word_rspace_removed =='×'))

# %%
vol1_char_df['potential_genus_match'] = vol1_char_df.apply(potential_genus_match, axis = 1)
vol1_char_df['potential_epithet_match'] = vol1_char_df.apply(potential_epithet_match, axis = 1)

vol2_char_df['potential_genus_match'] = vol2_char_df.apply(potential_genus_match, axis = 1)
vol2_char_df['potential_epithet_match'] = vol2_char_df.apply(potential_epithet_match, axis = 1)

vol3_char_df['potential_genus_match'] = vol3_char_df.apply(potential_genus_match, axis = 1)
vol3_char_df['potential_epithet_match'] = vol3_char_df.apply(potential_epithet_match, axis = 1)

# %%
all_vol_data_GE_potential_match_test = [(vol1_char_df, vol1_index, vol1_doc, "GE_potential_match_vol1"),
                                        (vol2_char_df, vol2_index, vol2_doc, "GE_potential_match_vol2"),
                                        (vol3_char_df, vol3_index, vol3_doc, "GE_potential_match_vol3")]

for vol_char_df, vol_index, doc, output_name in all_vol_data_GE_potential_match_test: 
    #for each volume 
    image_list = []

    for page_num in tqdm(vol_index):
        pix_map = doc.get_page_pixmap(page_num,matrix=mat)
        image = Image.open(io.BytesIO(pix_map.tobytes()))
        draw = ImageDraw.Draw(image)
        
        genus_db = vol_char_df[(vol_char_df['page_num'] == page_num) & 
                                     (vol_char_df['potential_genus_match'] == True)
                            ].loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin",	"char_bbox"])
                            ].drop_duplicates()

        epithet_db = vol_char_df[(vol_char_df['page_num'] == page_num) 
                                & (vol_char_df['potential_epithet_match'] == True)
                                ].loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin",	"char_bbox"])
                                ].drop_duplicates()

        #genus Coord is orange-pinkish, 5
        for coord in genus_db['word_bbox'] :
            x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
            draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#FF7F50"), width=5)
            
        # #epithet is red, 3
        for coord in epithet_db['word_bbox'] :
            x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
            draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#000099"), width=3)
        image_list.append(image)

    #save pages of the volume
    image_list[0].save('../output/local/'+output_name+'.pdf' ,save_all=True, append_images=image_list[1:])

# %% [markdown]
# ### SOME HARDCODING PARTS:

# %% [markdown]
# Erianthus

# %%
vol1_char_df[vol1_char_df['word'].str.contains('d,IlLIlU')]

# %% [markdown]
# based on this image output in volumen 1:
#  ![Erianthus](Erianthus.png)
# 
# 

# %%
weird_old_char_vol1 = vol1_char_df.loc[1731810:1731830]
weird_old_char_vol1['word']

# %%
vol_num_Erianthus1, page_num_Erianthus1, block_num_Erianthus1, line_num_Erianthus1, span_num_Erianthus1, word_num_Erianthus1 = vol1_char_df[vol1_char_df['word'].str.contains('J_JI')][['vol_num', 'page_num', 'block_num', 'line_num', 'span_num', 'word_num']].iloc[0]

# %%
vol1_char_df.columns

# %%
vol_num_Erianthu2, page_num_Erianthus2, block_num_Erianthus2, line_num_Erianthus2, span_num_Erianthus2, word_num_Erianthus2 = vol1_char_df[vol1_char_df['word'].str.contains('J.d,IlLIlU.')][['vol_num', 'page_num', 'block_num', 'line_num', 'span_num','word_num']].iloc[0]

# %%
target_index = vol1_char_df[((vol1_char_df['vol_num'] == vol_num_Erianthus1) | (vol1_char_df['vol_num'] == vol_num_Erianthu2)) &
                            ((vol1_char_df['page_num'] == page_num_Erianthus1) | (vol1_char_df['page_num'] == page_num_Erianthus2)) &
                            ((vol1_char_df['block_num'] == block_num_Erianthus1) | (vol1_char_df['block_num'] == block_num_Erianthus2)) & 
                            ((vol1_char_df['line_num'] == line_num_Erianthus1) | (vol1_char_df['line_num'] == line_num_Erianthus2)) &  
                            ((vol1_char_df['span_num'] == span_num_Erianthus1) | (vol1_char_df['span_num'] == span_num_Erianthus2)) & 
                            ((vol1_char_df['word_num'] == word_num_Erianthus1) | (vol1_char_df['word_num'] == word_num_Erianthus2)) 
                            ].index

# %%
start_index = 1731813
end_index = 1731827
weird_old_char_vol1 = vol1_char_df.loc[target_index]
weird_old_char_vol1['word']

# %% [markdown]
# make comment about this part is hard coded thingi

# %%
weird_old_char_vol1['word_num']

# %%
#manually fixing the OCR error for J_JI J.d,IlLIlU. hostii Griseb.
vol1_char_df.loc[target_index, 'word'] = 'Erianthus'
vol1_char_df.loc[target_index, 'word_num'] = 0
vol1_char_df.loc[target_index, 'pruned_word'] = 'Erianthus'
temp_word_x0 = vol1_char_df.loc[target_index, 'word_bbox'].apply(lambda x : x[0]).min()
temp_word_y0 = vol1_char_df.loc[target_index, 'word_bbox'].apply(lambda x : x[1]).min()
temp_word_x1 = vol1_char_df.loc[target_index, 'word_bbox'].apply(lambda x : x[2]).max()
temp_word_y1 = vol1_char_df.loc[target_index, 'word_bbox'].apply(lambda x : x[3]).max()
vol1_char_df.loc[target_index, 'word_bbox'] =vol1_char_df.loc[target_index, 'word_bbox'].apply(lambda x : (temp_word_x0, temp_word_y0, temp_word_x1, temp_word_y1))

vol1_char_df.loc[target_index, 'potential_epithet_match'] = False
vol1_char_df.loc[target_index, 'potential_genus_match'] = True

# %% [markdown]
# ### Infra species

# %%
# Reminder:
# all_vol_data_coord_match = [(vol1_char_df, vol1_index),
#                             (vol2_char_df, vol2_index),
#                             (vol3_char_df, vol3_index)]
for vol_char_df, vol_index in all_vol_data_coord_match: 
    vol_char_df["infra_coord_match"] = vol_char_df["word_bbox"].apply(lambda x : False)
    for page_num in tqdm(vol_index):

        margin = 1.25 * vol_char_df[(vol_char_df["potential_epithet_match"] == True) | (vol_char_df["potential_genus_match"] == True)]["char_bbox"].apply(lambda x : x[2] - x[0]).mean()
        
        mean_left_epithet = vol_char_df[(vol_char_df["page_num"] == page_num) & (vol_char_df["col_num"] == 0) & (vol_char_df["potential_epithet_match"] == True)]["word_bbox"].apply(lambda x : x[0]).mean()
        mean_left_genus = vol_char_df[(vol_char_df["page_num"] == page_num) & (vol_char_df["col_num"] == 0) & (vol_char_df["potential_genus_match"] == True)]["word_bbox"].apply(lambda x : x[0]).mean()
        if math.isnan(mean_left_genus):
            mean_left_genus_all = vol_char_df[(vol_char_df["col_num"] == 0) & (vol_char_df["potential_genus_match"] == True)]["word_bbox"].apply(lambda x : x[0]).mean()
            mean_left_epithet_all = vol_char_df[(vol_char_df["col_num"] == 0) & (vol_char_df["potential_epithet_match"] == True)]["word_bbox"].apply(lambda x : x[0]).mean()
            mean_left_tab = mean_left_epithet_all - mean_left_genus_all
        else: 
            mean_left_tab = mean_left_epithet - mean_left_genus
        
        mean_right_epithet = vol_char_df[(vol_char_df["page_num"] == page_num) & (vol_char_df["col_num"] == 1) & (vol_char_df["potential_epithet_match"] == True)]["word_bbox"].apply(lambda x : x[0]).mean()
        mean_right_genus = vol_char_df[(vol_char_df["page_num"] == page_num) & (vol_char_df["col_num"] == 1) & (vol_char_df["potential_genus_match"] == True)]["word_bbox"].apply(lambda x : x[0]).mean()
        if math.isnan(mean_right_genus):
            mean_right_genus_all = vol_char_df[(vol_char_df["col_num"] == 1) & (vol_char_df["potential_genus_match"] == True)]["word_bbox"].apply(lambda x : x[0]).mean()
            mean_right_epithet_all = vol_char_df[(vol_char_df["col_num"] == 1) & (vol_char_df["potential_epithet_match"] == True)]["word_bbox"].apply(lambda x : x[0]).mean()
            mean_right_tab = mean_right_epithet_all - mean_right_genus_all
        else: 
            mean_right_tab = mean_right_epithet - mean_right_genus


        vol_char_df.loc[(vol_char_df["page_num"] == page_num) & (vol_char_df["word_num"] == 0)  , "infra_coord_match"] = vol_char_df[(vol_char_df["page_num"] == page_num) & (vol_char_df["word_num"] == 0)]["word_bbox"].apply(lambda x : is_coord_match(x, mean_left_epithet + mean_left_tab, mean_right_epithet + mean_right_tab, margin))

# %%
# Takes longer but makes more sense generally. We will skip it here
# def potential_author_match_infra_coord(row):
#     word = row['word']
#     pruned_word = row['pruned_word']
#     lower_word = word.lower()
#     latin_connectives = r"^\s?et[\s|.]?$|^\s?in[\s|.]?$|^\s?non[\s|.]?$|^\s?&[\s|.]?$|^\s?er[\s|.]?$|^\s?nec[\s|.]?$|^\s?mult[\s|.]?$|^\s?ex[\s|.]?$|^\s?fil[\s|.]?$"
#     infra_symbols = r"^var[\s|.|\b]?$|^subsp[\s|.|\b]?$|^ssp[\s|.|\b]?$|^spp[\s|.|\b]?$|^x[\s|.|\b]?$|^×[\s|.|\b]?$"
#     is_latin_connectives = re.search(latin_connectives, word) != None
#     is_infra_symbol = re.search(infra_symbols, lower_word) != None
#     if pruned_word:
#         is_upper_first = pruned_word[0].isupper()
#     else:
#         is_upper_first = False
#     return (not is_infra_symbol) and (is_upper_first or is_latin_connectives)

def potential_author_match_infra_coord(word):
    lower_word = word.lower()
    latin_connectives = r"^\s?et[\s|.]?$|^\s?in[\s|.]?$|^\s?non[\s|.]?$|^\s?&[\s|.]?$|^\s?er[\s|.]?$|^\s?nec[\s|.]?$|^\s?mult[\s|.]?$|^\s?ex[\s|.]?$|^\s?fil[\s|.]?$"
    infra_symbols = r"^var[\s|.|\b]?$|^subsp[\s|.|\b]?$|^ssp[\s|.|\b]?$|^spp[\s|.|\b]?$|^x[\s|.|\b]?$|^×[\s|.|\b]?$"
    is_latin_connectives = re.search(latin_connectives, word) != None
    is_infra_symbol = re.search(infra_symbols, lower_word) != None
    return (not is_infra_symbol) and (word[0].isupper() or is_latin_connectives)

# %%
def has_infra_symbols(word):
    infra_symbols = r"^var[\s|.|\b]?$|^subsp[\s|.|\b]?$|^ssp[\s|.|\b]?$|^spp[\s|.|\b]?$|^x[\s|.|\b]?$|^×[\s|.|\b]?$"
    return re.search(infra_symbols, word) != None

# %%
# Reminder:
# all_vol_data_coord_match = [(vol1_char_df, vol1_index),
#                             (vol2_char_df, vol2_index),
#                             (vol3_char_df, vol3_index)]
for vol_char_df, _ in all_vol_data_coord_match:
    vol_char_df["potential_infra_match"] = (vol_char_df['word'].apply(has_infra_symbols)) | \
                                           ((vol_char_df["infra_coord_match"] == True) & (vol_char_df['word'].apply(potential_author_match_infra_coord) == False))

# %%
all_vol_dat_infra_match_test = [(vol1_char_df, vol1_index, vol1_doc, "potential_infra_match_vol1"),
                                (vol2_char_df, vol2_index, vol2_doc, "potential_infra_match_vol2"),
                                (vol3_char_df, vol3_index, vol3_doc, "potential_infra_match_vol3")][::-1]

for vol_char_df, vol_index, doc, output_name in all_vol_dat_infra_match_test: 
    #for each volume 
    image_list = []

    for page_num in tqdm(vol_index):
        pix_map = doc.get_page_pixmap(page_num,matrix=mat)
        image = Image.open(io.BytesIO(pix_map.tobytes()))
        draw = ImageDraw.Draw(image)
        

        infra_coord_db = vol_char_df[(vol_char_df['page_num'] == page_num) & 
                                     (vol_char_df['infra_coord_match'] == True)
                            ].loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin",	"char_bbox"])
                            ].drop_duplicates()

        infra_db = vol_char_df[(vol_char_df['page_num'] == page_num) 
                                & (vol_char_df['potential_infra_match'] == True)
                                ].loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin",	"char_bbox"])
                                ].drop_duplicates()

        with_infra_symbols = vol_char_df[(vol_char_df['page_num'] == page_num) &
                                         (vol_char_df['infra_coord_match'] == True) & 
                                         (vol_char_df['word'].apply(has_infra_symbols) == True)
                                        ].loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin",	"char_bbox"])
                                        ].drop_duplicates()

        #genus Coord is orange-pinkish, 5
        for coord in infra_coord_db['word_bbox'] :
            x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
            draw.rectangle((x0-5, y0-5, x1+5, y1+5), fill=None, outline=ImageColor.getrgb("#003399"), width=7)

        for coord in infra_db['word_bbox'] :
            x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
            draw.rectangle((x0-3, y0-3, x1+3, y1+3), fill=None, outline=ImageColor.getrgb("#FF7F50"), width=5)
            
        # #epithet is red, 3
        for coord in with_infra_symbols['word_bbox'] :
            x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
            draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#990000"), width=3)

        image_list.append(image)

    #save pages of the volume
    image_list[0].save('../output/local/'+output_name+'.pdf' ,save_all=True, append_images=image_list[1:])

# %% [markdown]
# ### page num processings

# %%
vol1_char_df['index_page_num'] = vol1_char_df['page_num'] - vol1_index[0] + 1
vol2_char_df['index_page_num'] = vol2_char_df['page_num'] - vol2_index[0] + 1
vol3_char_df['index_page_num'] = vol3_char_df['page_num'] - vol3_index[0] + 1

# %%
# all_vol_data_col_num = [(vol1_char_df, vol1_index, vol1_doc),
#                         (vol2_char_df, vol2_index, vol2_doc),
#                         (vol3_char_df, vol3_index, vol3_doc)]

for vol_char_df ,vol_index, vol_doc in all_vol_data_col_num: 
    #for each volume check if genus pattern / epithet pattern exists within the index part of the book
    for page_num in tqdm(vol_index):
        center_x0 = get_center_x0(vol_char_df, page_num, - 30)
        #find center based on x0 coordinate of each line
        vol_char_df['col_num_for_PN'] = vol_char_df['line_bbox'].apply(lambda coords : get_col_num(coords, center_x0)) 

# %%
def is_page_num(row):
    return row['pruned_word'].isnumeric()

# %%
# Reminder:
# all_vol_data_coord_match = [(vol1_char_df, vol1_index),
#                             (vol2_char_df, vol2_index),
#                             (vol3_char_df, vol3_index)]
for vol_char_df, vol_index in all_vol_data_coord_match: 
    vol_char_df['page_num_index_pat_match'] = (vol_char_df['page_num'].isin(vol_index)) & (vol_char_df.apply(is_page_num, axis = 1))
    vol_char_df["page_num_coord_match"] = vol_char_df["word_bbox"].apply(lambda x : False)
    for page_num in tqdm(vol_index):
        margin = 1.25 * vol_char_df[(vol_char_df["page_num_index_pat_match"] == True)]["char_bbox"].apply(lambda x : x[2] - x[0]).mean()
        page_num_char_df = vol_char_df[(vol_char_df["page_num"] == page_num) & (vol_char_df["page_num_index_pat_match"] == True)]
        page_num_df = page_num_char_df.loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin", "char_bbox"])].drop_duplicates()
        page_page_num_2dic = [{}, {}]
        
        for i in range(page_num_df.shape[0]):
            e_index = str(page_num) + "_" + str(i)
            p0 = page_num_df['word_bbox'].iloc[i]
            x_ref = p0[2]
            col = page_num_df['col_num_for_PN'].iloc[i]

            ref_neighbors_df = page_num_df[(page_num_df["page_num"] == page_num) & 
                                           (page_num_df["word_bbox"].apply(lambda x : x_ref - margin <= x[2] and x[2] <= x_ref + margin))]
            
            num_neighbors = ref_neighbors_df.shape[0]
            mean_neighbors = ref_neighbors_df["word_bbox"].apply(lambda x : x[2]).mean()
            page_page_num_2dic[col][e_index] = (num_neighbors, mean_neighbors)
        
        mean_left_page_num = max(page_page_num_2dic[0].values(), default = [-1, -1])[1]
        mean_right_page_num = max(page_page_num_2dic[1].values(), default = [-1, -1])[1]

        if mean_left_page_num == -1 or mean_right_page_num == -1:
            mean_valid_col = max(mean_left_page_num, mean_right_page_num)
            vol_char_df.loc[(vol_char_df["page_num"] == page_num) , "page_num_coord_match"] = vol_char_df[(vol_char_df["page_num"] == page_num)]["pruned_word_bbox"].apply(lambda x : is_coord_match([x[2]], mean_valid_col, mean_valid_col, margin))
        elif mean_left_page_num == -1 and mean_right_page_num == -1:
            vol_char_df.loc[(vol_char_df["page_num"] == page_num) , "page_num_coord_match"] = vol_char_df[(vol_char_df["page_num"] == page_num)]["pruned_word_bbox"].apply(lambda x : False)
        else: 
            vol_char_df.loc[(vol_char_df["page_num"] == page_num) , "page_num_coord_match"] = vol_char_df[(vol_char_df["page_num"] == page_num)]["pruned_word_bbox"].apply(lambda x : is_coord_match([x[2]], mean_left_page_num, mean_right_page_num, margin))


# %%
all_vol_data_PN_test = [(vol1_char_df, vol1_index, vol1_doc, "potential_page_num_match_vol1"),
                        (vol2_char_df, vol2_index, vol2_doc, "potential_page_num_match_vol2"),
                        (vol3_char_df, vol3_index, vol3_doc, "potential_page_num_match_vol3")][::-1]

for vol_char_df, vol_index, doc, output_name in all_vol_data_PN_test: 
    #for each volume 
    image_list = []

    for page_num in tqdm(vol_index):
        pix_map = doc.get_page_pixmap(page_num,matrix=mat)
        image = Image.open(io.BytesIO(pix_map.tobytes()))
        draw = ImageDraw.Draw(image)
        

        page_num_coord_db = vol_char_df[(vol_char_df['page_num'] == page_num) & 
                                     (vol_char_df['page_num_coord_match'] == True)
                            ].loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin",	"char_bbox"])
                            ].drop_duplicates()

        # infra_db = vol_char_df[(vol_char_df['page_num'] == page_num) 
        #                         & (vol_char_df['potential_infra_match'] == True)
        #                         ].loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin",	"char_bbox"])
        #                         ].drop_duplicates()

        # with_infra_symbols = vol_char_df[(vol_char_df['page_num'] == page_num) &
        #                                  (vol_char_df['infra_coord_match'] == True) & 
        #                                  (vol_char_df['word'].apply(has_infra_symbols) == True)
        #                                 ].loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin",	"char_bbox"])
        #                                 ].drop_duplicates()

        #genus Coord is orange-pinkish, 5
        for coord in page_num_coord_db['word_bbox'] :
            x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
            draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#003399"), width=3)

        # for coord in infra_db['word_bbox'] :
        #     x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
        #     draw.rectangle((x0-3, y0-3, x1+3, y1+3), fill=None, outline=ImageColor.getrgb("#FF7F50"), width=5)
            
        # # #epithet is red, 3
        # for coord in with_infra_symbols['word_bbox'] :
        #     x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
        #     draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#990000"), width=3)

        image_list.append(image)

    #save pages of the volume
    image_list[0].save('../output/local/'+output_name+'.pdf' ,save_all=True, append_images=image_list[1:])

# %% [markdown]
# ### catching & hardcoding issues

# %% [markdown]
# #### checking potential_infra_match that are not with typical symbols
# 

# %%
vol1_char_df[(vol1_char_df['potential_infra_match'] == True) & (vol1_char_df['word'].apply(has_infra_symbols) == False)][["index_page_num", "word"]].drop_duplicates()

# %%
vol2_char_df[(vol2_char_df['potential_infra_match'] == True) & (vol2_char_df['word'].apply(has_infra_symbols) == False)][["index_page_num", "word"]].drop_duplicates()

# %%
vol3_char_df[(vol3_char_df['potential_infra_match'] == True) & (vol3_char_df['word'].apply(has_infra_symbols) == False)][["index_page_num", "word"]].drop_duplicates()

# %%
remove_infra_index = [1566359, 1570483, 1578491, 1581443, 1582167, 1594835, 1598587]
#temp_df_hard_code_infra = vol3_char_df[(vol3_char_df['potential_infra_match'] == True) & (vol3_char_df['word'].apply(has_infra_symbols) == False)][["index_page_num", "pruned_word"]].drop_duplicates()
#temp_df_hard_code_infra[temp_df_hard_code_infra['pruned_word'].apply(lambda x : len(x) > 3)].index 
#nice ways but won't have everything in them ... so 

# %%
def get_index_end(vol_df, start_index):
    len_word = len(vol_df.loc[start_index,'word'])
    print(len_word, start_index + len_word)
    return start_index + len_word - 1

set_epithet_index = [1581443]
remove_infra_index = [1570483, 1566359, 1570883, 1570883, 1578491, 1581443, 1582167, 1594835, 1598587]
for i in remove_infra_index:
    vol3_char_df.loc[i : get_index_end(vol3_char_df, i),'potential_infra_match'] = False

for i in set_epithet_index:
    vol3_char_df.loc[i : get_index_end(vol3_char_df, i),'potential_epithet_match'] = True


# %%
#cock keeps needing to be pruned multiple times ....??? not sure why ugh
vol3_char_df[(vol3_char_df['potential_infra_match'] == True) & (vol3_char_df['word'].apply(has_infra_symbols) == False)][["index_page_num", "word"]].drop_duplicates()

# %% [markdown]
# 1596384 -> var with space in between it somehow

# %%
ending = get_index_end(vol3_char_df, 1596384)+2
vol3_char_df.loc[1596384 : ending, 'word'] = 'var.'
vol3_char_df.loc[1596384 : ending, 'word_num'] = 0
vol3_char_df.loc[1596384 : ending, 'pruned_word'] = 'var'
vol3_char_df.loc[1596384 : ending, 'potential_infra_match'] = True 
#have to run it again this is problematic now

# %% [markdown]
# #### upper case beggining / latin words in epithet coordd

# %%
def potential_author_match_epithet_coord(word):
    latin_connectives = r"^\s?et[\s|.]?$|^\s?in[\s|.]?$|^\s?non[\s|.]?$|^\s?&[\s|.]?$|^\s?er[\s|.]?$|^\s?nec[\s|.]?$|^\s?mult[\s|.]?$|^\s?ex[\s|.]?$|^\s?fil[\s|.]?$|^\s?f[\s|.]?$"
    is_latin_connectives = re.search(latin_connectives, word) != None
    is_hybrid = word == "X"
    return is_latin_connectives or (word[0].isupper() and (not is_hybrid))

# %% [markdown]
# CHECKED: for vol1 all are okay and are just typos 

# %%
vol1_char_df[(vol1_char_df['potential_epithet_match'] == True) & (vol1_char_df['word'].apply(potential_author_match_epithet_coord))][["index_page_num", "word"]].drop_duplicates()

# %% [markdown]
# CHECKED Vol2 
# 
# Hbanoticus -> typo for libanoticus 
# 
# Hppii -> typo for lippii
# 
# letting fuzzy matching take care of it later :)
# 
# **TODO**: Ma -> is supposed to be chia (not easy for fuzzy matching to fix...)
# 

# %%
vol2_char_df[(vol2_char_df['potential_epithet_match'] == True) & (vol2_char_df['word'].apply(potential_author_match_epithet_coord))][["index_page_num", "word"]].drop_duplicates()

# %% [markdown]
# CHECKED vol3 

# %%
vol3_char_df[(vol3_char_df['potential_epithet_match'] == True) & (vol3_char_df['word'].apply(potential_author_match_epithet_coord))][["index_page_num", "word"]].drop_duplicates()

# %%
# Eichwaldii, Krascheninnikovii, DOteriifolium (p should be p) -> oki 
# Majoranamaracus -> hybrid situation -> fixed later
not_epithet_index_list = [1565497, 1566524, 1575185, 1575488, 1577207, 1579044, 1582101, 1583001, 1586349, 1586394, 1592464, 1608442, 1609136, 1610970, 1611185]

for i in not_epithet_index_list:
    vol3_char_df.loc[i : get_index_end(vol3_char_df, i),'potential_epithet_match'] = False

# %%
vol3_char_df[(vol3_char_df['potential_epithet_match'] == True) & (vol3_char_df['word'].apply(potential_author_match_epithet_coord))][["index_page_num", "word"]].drop_duplicates()

# %% [markdown]
# #### [**TODO FIX LATER**] epithet coord word has uppper case in the middle (but not the first letter)

# %%
def has_upper_not_first(word):
    return word[1:].lower() != word[1:]

# %%
vol1_char_df[(vol1_char_df['potential_epithet_match'] == True) & (vol1_char_df['word'].apply(has_upper_not_first))][["index_page_num", "word"]].drop_duplicates()

# %%
vol2_char_df[(vol2_char_df['potential_epithet_match'] == True) & (vol2_char_df['word'].apply(has_upper_not_first))][["index_page_num", "word"]].drop_duplicates()

# %%
vol3_char_df[(vol3_char_df['potential_epithet_match'] == True) & (vol3_char_df['word'].apply(has_upper_not_first))][["index_page_num", "word"]].drop_duplicates()

# %% [markdown]
# #### hardcoding Genus Platanthera

# %%
vol1_char_df[(vol1_char_df['potential_genus_match'] == True) & (vol1_char_df['word'] == 'Platan')][['word_num', 'char_num']]

# %%
vol1_char_df.shape[0]

# %%
vol1_char_df[(vol1_char_df['potential_genus_match'] == True) & (vol1_char_df['word'] == 'Platan')][['page_num', 'block_num','line_num']].iloc[0]

# %%
vol1_char_df[(vol1_char_df['page_num'] == 631) & (vol1_char_df['block_num'] == 87) & (vol1_char_df['line_num'] == 0)][['page_num', 'block_num','line_num', 'word', 'word_num', 'char', 'span_num']]

# %%
vol1_char_df.loc[(vol1_char_df['page_num'] == 631) & (vol1_char_df['block_num'] == 87) & (vol1_char_df['line_num'] == 0),'potential_genus_match'] = True
vol1_char_df.loc[(vol1_char_df['page_num'] == 631) & (vol1_char_df['block_num'] == 87) & (vol1_char_df['line_num'] == 0),'potential_epithet_match'] = False
vol1_char_df.loc[(vol1_char_df['page_num'] == 631) & (vol1_char_df['block_num'] == 87) & (vol1_char_df['line_num'] == 0),'word_num'] = 0
vol1_char_df.loc[(vol1_char_df['page_num'] == 631) & (vol1_char_df['block_num'] == 87) & (vol1_char_df['line_num'] == 0),'word'] = 'Platanthera'
vol1_char_df.loc[(vol1_char_df['page_num'] == 631) & (vol1_char_df['block_num'] == 87) & (vol1_char_df['line_num'] == 0),'pruned_word'] = 'Platanthera'

# %%
vol1_char_df.loc[1746680, 'word']

# %%
# keep_cols = vol1_char_df.columns.difference(["char_num", "char", "char_origin", "char_bbox"])
# first_entry = vol1_char_df.loc[(vol1_char_df['page_num'] == 631) & (vol1_char_df['block_num'] == 87) & (vol1_char_df['line_num'] == 0)].iloc[0]
# for col_name in keep_cols: 
#     print(vol1_char_df.loc[(vol1_char_df['page_num'] == 631) & (vol1_char_df['block_num'] == 87) & (vol1_char_df['line_num'] == 0), col_name], first_entry[col_name])
#     vol1_char_df.loc[(vol1_char_df['page_num'] == 631) & (vol1_char_df['block_num'] == 87) & (vol1_char_df['line_num'] == 0), col_name] = first_entry[col_name]
#     #print(first_entry[col_name])

# %%
vol1_char_df[(vol1_char_df['page_num'] == 631) & (vol1_char_df['block_num'] == 87) & (vol1_char_df['line_num'] == 0)][['page_num', 'block_num','line_num', 'word', 'word_num', 'char', 'char_num']]

# %%
# make same 
# iterate -> char_num = list(range(df.shape[0]))
# get max 
# get min 

# %%
def copy_data(df, src_index, dst_indecies, col_name, inplace = True):
    """ used for modifying columns of datatype that can be dirrectly assigned to multiple rows
        INPUT: 
              - df: target dataframe to modify 
              - 
        OUTPUT: returns df 
    """
    if inplace == False:
        df = df.copy()
    
    src_value = df.loc[src_index, col_name]
    if isinstance(src_value, tuple):
        df.loc[dst_indecies, col_name] = [src_value] * len(dst_indecies)
    else: 
        df.loc[dst_indecies, col_name] = src_value

    return df 

# %%
def copy_tuple_data(df, src_index, dst_indecies, col_name, inplace = True):
    """ Used for modifying columns with lists/tuples
        INPUT: 
              - df: target dataframe to modify 
              - 
        OUTPUT: returns df 
    """
    if inplace == False:
        df = df.copy()
    
    src_value = df.loc[src_index, col_name]
    df.loc[dst_indecies, col_name] = [src_value] * len(dst_indecies)

    return df 

# %%
def reiterate(df, src_index, dst_indecies, col_name, inplace = True):
    """ Used for modifying columns with lists/tuples
        INPUT: 
              - df: target dataframe to modify 
              - 
        OUTPUT: returns df 
    """
    if inplace == False:
        df = df.copy()
    
    df.loc[dst_indecies, col_name] = list(range(len(dst_indecies)))

    return df 

# %%
def transform_values(df, src_index, dst_indecies, col_name, transformations, inplace = True):
    """ Used for modifying columns with lists/tuples
        INPUT: 
              - df: target dataframe to modify 
              - 
        OUTPUT: returns df 
    """
    if inplace == False:
        df = df.copy()

    # if src_value is of type list or tuple or something like that, make sure transformations are available for each index or the size is 1. 
    # the apply transformations to each index
    

    df.loc[dst_indecies, col_name]

    src_value = df.loc[src_index, col_name]
    
    df.loc[dst_indecies, col_name] = src_value

    return df 

# %%
def merge_bbox(df, indecies, col_name, inplace = True):
    """ Used for modifying columns with lists/tuples
        INPUT: 
              - df: target dataframe to modify 
              - 
        OUTPUT: returns df 
    """
    if inplace == False:
        df = df.copy()

    # if src_value is of type list or tuple or something like that, make sure transformations are available for each index or the size is 1. 
    # the apply transformations to each index
    

    x0 = df.loc[indecies, col_name].apply(lambda x: x[0]).min()
    y0 = df.loc[indecies, col_name].apply(lambda x: x[1]).min()
    x1 = df.loc[indecies, col_name].apply(lambda x: x[2]).max()
    y1 = df.loc[indecies, col_name].apply(lambda x: x[3]).max()
    

    new_bbox = (x0, y0, x1, y1)
    df.loc[indecies, col_name] = [new_bbox] * len(indecies)

    return df 

# %%
def merge_str(df, src_index, dst_indecies, col_name, inplace = True):
    
    return 


# %%
vol1_char_df.loc[100, 'word']

# %%
# need to fix anything vol_index_df.columns.difference(["char_num", "char", "char_origin", "char_bbox"]

# %% [markdown]
# #### Potential genus mismatch

# %% [markdown]
# potential genus match but name is not alphabetic or is of length < 3

# %%
def flag_genus_name(word):
    word_no_space = word.replace(" ", "")
    return ((not word_no_space.isalpha()) or (len(word_no_space) < 3))

# %%
vol1_char_df[(vol1_char_df['potential_genus_match'] == True) & (vol1_char_df['word'].apply(flag_genus_name))][["index_page_num", "word"]].drop_duplicates()
#skipping over all these 

# %%
not_genus_index_list_vol1 = [1730943, 1738656, 1754704]
for i in not_genus_index_list_vol1:
    vol1_char_df.loc[i : get_index_end(vol1_char_df, i),'potential_genus_match'] = False

# %%
vol1_char_df[(vol1_char_df['potential_genus_match'] == True) & (vol1_char_df['word'].apply(flag_genus_name))][["index_page_num", "word"]].drop_duplicates()

# %%
vol2_char_df[(vol2_char_df['potential_genus_match'] == True) & (vol2_char_df['word'].apply(flag_genus_name))][["index_page_num", "word"]].drop_duplicates()

# %%
not_genus_index_list_vol2 = [1939606]
for i in not_genus_index_list_vol2:
    vol2_char_df.loc[i : get_index_end(vol2_char_df, i),'potential_genus_match'] = False

# %%
vol2_char_df[(vol2_char_df['potential_genus_match'] == True) & (vol2_char_df['word'].apply(flag_genus_name))][["index_page_num", "word"]].drop_duplicates()

# %%
vol3_char_df[(vol3_char_df['potential_genus_match'] == True) & (vol3_char_df['word'].apply(flag_genus_name))][["index_page_num", "word"]].drop_duplicates()
#all okay and hybrid will get fixed later :)

# %%
vol1_char_df[(vol1_char_df['potential_genus_match'] == True) & (vol1_char_df['word'].apply(lambda x : x[0].isupper() == False))]['word']

# %%
index_apetala_correcting = vol1_char_df[(vol1_char_df['potential_genus_match'] == True) & (vol1_char_df['word'].apply(lambda x : x[0].isupper() == False))].index
vol1_char_df.loc[index_apetala_correcting,'potential_genus_match'] = False
vol1_char_df.loc[index_apetala_correcting,'potential_epithet_match'] = True

# %%
vol1_char_df[(vol1_char_df['potential_genus_match'] == True) & (vol1_char_df['word'].apply(lambda x : x[0].isupper() == False))]['word']

# %%
#fuzzy matching will fix
vol2_char_df[(vol2_char_df['potential_genus_match'] == True) & (vol2_char_df['word'].apply(lambda x : x[0].isupper() == False))]['word']

# %%
#fixed later when hybrids identified 
vol3_char_df[(vol3_char_df['potential_genus_match'] == True) & (vol3_char_df['word'].apply(lambda x : x[0].isupper() == False))]['word']

# %% [markdown]
# ### pruning char_df and getting index_df

# %%
[c for c in vol1_char_df.columns if c.startswith('potential')]

# %%
#making sure page_num is in index
#making sure the genus level word is not all uppercase (a family name)
#making sure the pruned_word is not numeric (removing page_number as it's not in order usually) and removing page_num_coord_match

all_vol_data = [(vol1_char_df, vol1_index),
                (vol2_char_df, vol2_index),
                (vol3_char_df, vol3_index)]

result = [] 
ignore_word_list = ["NOUVELLE", "Flore", "FLORE", "INDEX", ""]
for vol_char_df, vol_index in all_vol_data:
    curr_result_df = vol_char_df[(vol_char_df['page_num'].isin(vol_index)) &
                                (~((vol_char_df["word"].str.isupper()) & (vol_char_df["word"].apply(lambda x : len(x) > 2)) & (vol_char_df['genus_coord_match'] == True))) & 
                                (~(vol_char_df["pruned_word"].isin(ignore_word_list))) &
                                (~(vol_char_df["pruned_word"].str.isnumeric() & (vol_char_df["word"] != "(3"))) & 
                                (~(vol_char_df["page_num_coord_match"] == True)) & 
                                (~((vol_char_df.groupby(['page_num', 'block_num', 'line_num'])['char_num'].transform('max') == 0) & (vol_char_df['word'].str.isupper())))
                                ].copy()
    result.append(curr_result_df)

vol1_index_df, vol2_index_df, vol3_index_df = result[0], result[1], result[2]

# %%
all_vol_data_PN_test = [(vol1_index_df, vol1_index, vol1_doc, "valid_words_vol1"),
                        (vol2_index_df, vol2_index, vol2_doc, "valid_words_vol2"),
                        (vol3_index_df, vol3_index, vol3_doc, "valid_words_vol3")]

for vol_char_df, vol_index, doc, output_name in all_vol_data_PN_test: 
    #for each volume 
    image_list = []

    for page_num in tqdm(vol_index):
        pix_map = doc.get_page_pixmap(page_num,matrix=mat)
        image = Image.open(io.BytesIO(pix_map.tobytes()))
        draw = ImageDraw.Draw(image)
        temp_coords = vol_char_df[vol_char_df['page_num'] == page_num]['word_bbox'].drop_duplicates()
        for coord in temp_coords:
            x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
            draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#003399"), width=3)

        image_list.append(image)

    #save pages of the volume
    image_list[0].save('../output/local/'+output_name+'.pdf' ,save_all=True, append_images=image_list[1:])

# %%
#only keeping word level
vol_index_df_list = [vol1_index_df, vol2_index_df, vol3_index_df]
result_df = []
for vol_index_df in vol_index_df_list:
    keep_cols = vol_index_df.columns.difference(["char_num", "char", "char_origin", "char_bbox"], sort=False).tolist()

    vol_index_df = vol_index_df.copy().loc[:,keep_cols].drop_duplicates().reset_index()
    vol_index_df.rename(columns={"index": "char_index"}, inplace = True)
    result_df.append(vol_index_df)

vol1_index_df, vol2_index_df, vol3_index_df = result_df[0], result_df[1], result_df[2]


# %%
def has_hybrid_symbols(word):
    infra_symbols = r"^X[\s|.|\b]?$|^x[\s|.|\b]?$|^×[\s|.|\b]?$"
    return re.search(infra_symbols, word) != None

# %%
result_df_hybrids = []
for vol_index_df in [vol1_index_df, vol2_index_df, vol3_index_df]:
    vol_index_df['is_hybrid'] = np.NaN
    vol_index_df.loc[(vol_index_df['potential_infra_match'] == True) | (vol_index_df['potential_epithet_match'] == True) | (vol_index_df['potential_genus_match'] == True), 'is_hybrid'] = (vol_index_df['word'].apply(has_hybrid_symbols) == True) & ((vol_index_df['potential_infra_match'] == True) | (vol_index_df['potential_epithet_match'] == True) | (vol_index_df['potential_genus_match'] == True))
    
    hybrid_genera_indecies = vol_index_df[(vol_index_df['potential_genus_match'] == True) & (vol_index_df['word'].apply(has_hybrid_symbols) == True)].index + 1
    hybrid_epithet_indecies = vol_index_df[(vol_index_df['potential_epithet_match'] == True) & (vol_index_df['word'].apply(has_hybrid_symbols) == True)].index + 1
    
    vol_index_df.loc[hybrid_epithet_indecies, 'is_hybrid'] = True
    vol_index_df.loc[hybrid_epithet_indecies, 'potential_epithet_match'] = True 

    vol_index_df.loc[hybrid_genera_indecies, 'is_hybrid'] = True
    vol_index_df.loc[hybrid_genera_indecies, 'potential_genus_match'] = True

    drop_list = list(hybrid_epithet_indecies - 1) + list(hybrid_genera_indecies -1)
    
    vol_index_df = vol_index_df[~vol_index_df.index.isin(drop_list)].copy()
    #vol_index_df['is_hybrid'].ffill(inplace=True) fowrward fill after checking hybrid for the infra species types too

    result_df_hybrids.append(vol_index_df)

vol1_index_df, vol2_index_df, vol3_index_df = result_df_hybrids[0], result_df_hybrids[1], result_df_hybrids[2]

# %%
#df['closest_epithet_v2'] = np.nan
def extract_potential_genus_names(row):
    if row['potential_genus_match'] == True:
        return row['word'] + "_" + str(row['page_num']) + "_" + str(row['block_num']) + "_" + str(row['line_num'])
    else:
        return np.nan
        
for vol_index_df in [vol1_index_df, vol2_index_df, vol3_index_df]:
    vol_index_df['closest_genus'] = vol_index_df.apply(extract_potential_genus_names, axis = 1)
    vol_index_df['closest_genus'].ffill(inplace=True)

# %%
#df['closest_epithet_v2'] = np.nan
def extract_potential_epithet_names(row):
    if row['potential_epithet_match'] == True:
        return row['word'] + "_" + str(row['page_num']) + "_" + str(row['block_num']) + "_" + str(row['line_num'])
    else:
        return np.nan

for vol_index_df in [vol1_index_df, vol2_index_df, vol3_index_df]:
    vol_index_df['closest_epithet'] = vol_index_df.apply(extract_potential_epithet_names, axis = 1)
    vol_index_df.loc[vol_index_df['potential_genus_match'] == True, 'closest_epithet'] = -1
    vol_index_df['closest_epithet'].ffill(inplace=True)

# %%
def extract_potential_infra_type(row):
    if row['potential_infra_match'] == True:
        return row['word'] + "_" + str(row['page_num']) + "_" + str(row['block_num']) + "_" + str(row['line_num'])
    else:
        return np.nan

for vol_index_df in [vol1_index_df, vol2_index_df, vol3_index_df]:
    vol_index_df.loc[(vol_index_df['potential_epithet_match'] == True) | (vol_index_df['potential_genus_match'] == True), 'closest_infra_type'] = -1
    vol_index_df['closest_infra_type'] = vol_index_df.apply(extract_potential_infra_type, axis = 1)
    vol_index_df.loc[(vol_index_df['potential_infra_match'] == False) & ((vol_index_df['potential_epithet_match'] == True) | (vol_index_df['potential_genus_match'] == True)), 'closest_infra_type'] = -1
    vol_index_df['closest_infra_type'].ffill(inplace=True)

# %%
for vol_index_df in [vol1_index_df, vol2_index_df, vol3_index_df]:
    infra_name_match_indecies = vol_index_df[vol_index_df['potential_infra_match'] == True].index + 1
    vol_index_df['closest_infra_name'] = np.NaN
    vol_index_df.loc[infra_name_match_indecies, 'closest_infra_name'] = vol_index_df.apply(lambda row : row['word'] + "_" + str(row['page_num']) + "_" + str(row['block_num']) + "_" + str(row['line_num']) , axis = 1)
    vol_index_df['potential_infra_name_match'] = vol_index_df.index.isin(infra_name_match_indecies)
    vol_index_df.loc[(vol_index_df['potential_infra_match'] == True) | (vol_index_df['potential_epithet_match'] == True) | (vol_index_df['potential_genus_match'] == True), 'closest_infra_name'] = -1
    vol_index_df['closest_infra_name'].ffill(inplace=True)

# %%
for vol_index_df in [vol1_index_df, vol2_index_df, vol3_index_df]:
    cond = ((vol_index_df['closest_infra_name'] != '') | (vol_index_df['closest_infra_name'] != -1) | (~vol_index_df['closest_infra_name'].isna())) & \
           (vol_index_df['word'].apply(has_hybrid_symbols) == True)
    
    vol_index_df.loc[cond, 'is_hybrid'] = True
    vol_index_df['is_hybrid'].ffill(inplace=True)

# %%
for vol_index_df in [vol1_index_df, vol2_index_df, vol3_index_df]:
    vol_index_df['potential_author_match'] = (vol_index_df['potential_genus_match'] == False) & \
                                             (vol_index_df['potential_epithet_match'] == False) & \
                                             (vol_index_df['potential_infra_match'] == False) & \
                                             (vol_index_df['potential_infra_name_match'] == False)

# %%
for vol_index_df in [vol1_index_df, vol2_index_df, vol3_index_df]:
    vol_index_df.replace(-1, np.NaN, inplace = True)
    vol_index_df.replace(np.NaN, "",inplace = True)

# %%
#author grouping 
# 
author_grouping = ['closest_genus', 'closest_epithet', 'closest_infra_name']
merge_on = ['closest_genus', 'closest_epithet', 'closest_infra_name']
def concatenate(group):
    return group.loc[group['potential_author_match'] == True, 'word'].str.cat(sep=' ')

result_df_authors = [] 
for vol_index_df in [vol1_index_df, vol2_index_df, vol3_index_df]: 
    #author_grouping = ['closest_genus', 'closest_epithet']
    #merge_on = ['closest_genus', 'closest_epithet']
    groups = vol_index_df.groupby(author_grouping)
    concatenated = groups.apply(concatenate).reset_index()

    # add the concatenated values to the original dataframe
    result = vol_index_df.merge(concatenated[merge_on + [0]], on=merge_on, how='left').rename(columns={0: 'authors'})
    result_df_authors.append(result)
    
vol1_index_df, vol2_index_df, vol3_index_df = result_df_authors[0], result_df_authors[1], result_df_authors[2]


# %%
# for vol_index_df in [vol1_index_df, vol2_index_df, vol3_index_df]:
#     #vol_index_df.replace("", np.NaN,inplace = True)
#     vol_index_df.replace(np.NaN, "",inplace = True)

# %%
all_vol_data_cat_test = [(vol1_index_df, vol1_index, vol1_doc, "catagorized_vol1"),
                         (vol2_index_df, vol2_index, vol2_doc, "catagorized_vol2"),
                         (vol3_index_df, vol3_index, vol3_doc, "catagorized_vol3")]

for vol_char_df, vol_index, doc, output_name in all_vol_data_cat_test: 
    #for each volume 
    image_list = []

    for page_num in tqdm(vol_index):
        pix_map = doc.get_page_pixmap(page_num,matrix=mat)
        image = Image.open(io.BytesIO(pix_map.tobytes()))
        draw = ImageDraw.Draw(image)

        for col_num in [0, 1]:
            temp_df = vol_char_df[(vol_char_df['page_num'] == page_num) & (vol_char_df['col_num'] == col_num)]
            #genus Coord is orange-pinkish, 5
            for name, group in temp_df.groupby(['closest_genus'])['word_bbox']:
                x0 = (group.apply(lambda x : x[0]).min())*TARGET_DPI/ 72
                y0 = (group.apply(lambda x : x[1]).min())*TARGET_DPI/ 72
                x1 = (group.apply(lambda x : x[2]).max())*TARGET_DPI/ 72
                y1 = (group.apply(lambda x : x[3]).max())*TARGET_DPI/ 72
                draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#6939a3"), width=3)

            for name, group in temp_df.groupby(['closest_epithet'])['word_bbox']:
                if name != '':
                    x0 = (group.apply(lambda x : x[0]).min())*TARGET_DPI/ 72
                    y0 = (group.apply(lambda x : x[1]).min())*TARGET_DPI/ 72
                    x1 = (group.apply(lambda x : x[2]).max())*TARGET_DPI/ 72
                    y1 = (group.apply(lambda x : x[3]).max())*TARGET_DPI/ 72
                    draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#003399"), width=3)

            for name, group in temp_df.groupby(['closest_infra_name'])['word_bbox']:
                if name != '':
                    x0 = (group.apply(lambda x : x[0]).min())*TARGET_DPI/ 72
                    y0 = (group.apply(lambda x : x[1]).min())*TARGET_DPI/ 72
                    x1 = (group.apply(lambda x : x[2]).max())*TARGET_DPI/ 72
                    y1 = (group.apply(lambda x : x[3]).max())*TARGET_DPI/ 72
                    draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#8c690b"), width=3)

            temp_df_author_only = temp_df[temp_df['potential_author_match'] == True]
            for name, group in temp_df_author_only.groupby(['closest_genus', 'closest_epithet', 'closest_infra_name'])['word_bbox']:
                x0 = (group.apply(lambda x : x[0]).min())*TARGET_DPI/ 72
                y0 = (group.apply(lambda x : x[1]).min())*TARGET_DPI/ 72
                x1 = (group.apply(lambda x : x[2]).max())*TARGET_DPI/ 72
                y1 = (group.apply(lambda x : x[3]).max())*TARGET_DPI/ 72

                draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#9e9e9e"), width=3)


        image_list.append(image)

    #save pages of the volume
    image_list[0].save('../output/local/'+output_name+'.pdf' ,save_all=True, append_images=image_list[1:])

# %%
def fix_words(word):
    head, sep, tail = word.partition('_')
    return head 

for vol_index_df in [vol1_index_df, vol2_index_df, vol3_index_df]:
    vol_index_df['closest_genus'] = vol_index_df['closest_genus'].apply(fix_words)
    vol_index_df['closest_epithet'] = vol_index_df['closest_epithet'].apply(fix_words)
    vol_index_df['closest_infra_type'] = vol_index_df['closest_infra_type'].apply(fix_words)
    vol_index_df['closest_infra_name'] = vol_index_df['closest_infra_name'].apply(fix_words)

# %%
result_prune_authors_list = []
for vol_index_df in [vol1_index_df, vol2_index_df, vol3_index_df]:
    result_prune_authors = vol_index_df[(vol_index_df['potential_genus_match'] == True) |
                                        (vol_index_df['potential_epithet_match'] == True) |
                                        (vol_index_df['potential_infra_name_match'] == True)].copy()
    result_prune_authors_list.append(result_prune_authors)

prune_authors_vol1, prune_authors_vol2, prune_authors_vol3 =  result_prune_authors_list[0], result_prune_authors_list[1], result_prune_authors_list[2]

# %%
#quick fix for vol1 epithet on the same line as author situation
cond = (prune_authors_vol1['closest_genus'] != '') & (prune_authors_vol1['closest_epithet'] == '') & (prune_authors_vol1['authors'] != '')
prune_authors_vol1.loc[cond, 'closest_epithet'] = prune_authors_vol1.loc[cond,'authors'].str.split().apply(lambda s: s[0])
prune_authors_vol1.loc[cond, 'authors'] = prune_authors_vol1.loc[cond,'authors'].str.split().apply(lambda s: " ".join(s[1:]))

# %%
def get_taxon_rank_specific(row):
    has_genus = (pd.isnull(row['closest_genus']) == False) & (row['closest_genus'] != "") & (row['closest_genus'] != -1)
    has_epithet = (pd.isnull(row['closest_epithet']) == False) & (row['closest_epithet'] != "") & (row['closest_epithet'] != -1)
    
    has_infra = (pd.isnull(row['closest_infra_name']) == False) & (row['closest_infra_name'] != "") & (row['closest_infra_name'] != -1)
    has_infra_type = (pd.isnull(row['closest_infra_type']) == False) & (row['closest_infra_type'] != "") & (row['closest_infra_type'] != -1)
    infra_type = row['closest_infra_type']
    is_infra_hybrid = has_hybrid_symbols(row['closest_infra_type']) == True
    if is_infra_hybrid:
        infra_type = "hybrid"
    
    is_hybrid  = row['is_hybrid'] == True
    prefix  = ""
    if is_hybrid:
        prefix = "hybrid "

    if has_infra or has_infra_type:
        return f"infra ({infra_type})"
    if has_epithet:
        return prefix + "epithet"
    if has_genus:
        return prefix + "genus"

def get_taxon_rank_general(row):
    has_genus = (pd.isnull(row['closest_genus']) == False) & (row['closest_genus'] != "") & (row['closest_genus'] != -1)
    has_epithet = (pd.isnull(row['closest_epithet']) == False) & (row['closest_epithet'] != "") & (row['closest_epithet'] != -1)
    has_infra = (pd.isnull(row['closest_infra_name']) == False) & (row['closest_infra_name'] != "") & (row['closest_infra_name'] != -1)
    has_infra_type = (pd.isnull(row['closest_infra_type']) == False) & (row['closest_infra_type'] != "") & (row['closest_infra_type'] != -1)
    
    if has_infra or has_infra_type:
        return "infra"
    if has_epithet:
        return "epithet"
    if has_genus:
        return "genus"

# %%
prune_authors_vol1['closest_genus'] = prune_authors_vol1['closest_genus'].str.replace("œ", "oe" )
prune_authors_vol2['closest_genus'] = prune_authors_vol2['closest_genus'].str.replace("œ", "oe" )
prune_authors_vol3['closest_genus'] = prune_authors_vol3['closest_genus'].str.replace("œ", "oe" )

prune_authors_vol1['closest_epithet'] = prune_authors_vol1['closest_epithet'].str.replace("œ", "oe" )
prune_authors_vol2['closest_epithet'] = prune_authors_vol2['closest_epithet'].str.replace("œ", "oe" )
prune_authors_vol3['closest_epithet'] = prune_authors_vol3['closest_epithet'].str.replace("œ", "oe" )

prune_authors_vol1['closest_infra_name'] = prune_authors_vol1['closest_infra_name'].str.replace("œ", "oe" )
prune_authors_vol2['closest_infra_name'] = prune_authors_vol2['closest_infra_name'].str.replace("œ", "oe" )
prune_authors_vol3['closest_infra_name'] = prune_authors_vol3['closest_infra_name'].str.replace("œ", "oe" )

# %%
for vol_index_df in [prune_authors_vol1, prune_authors_vol2, prune_authors_vol3]:
    vol_index_df['taxon_rank'] = vol_index_df.apply(get_taxon_rank_general, axis = 1)
    vol_index_df['taxon_rank_detailed'] = vol_index_df.apply(get_taxon_rank_specific, axis = 1)

# %%
simplified_vol1 = prune_authors_vol1[['closest_genus',
                                      'closest_epithet',
                                      'closest_infra_name',
                                      'authors',
                                      'taxon_rank',
                                      'taxon_rank_detailed']]
simplified_vol1.to_csv('../output/local/index_output/vol1_index_output.csv')

simplified_vol2 = prune_authors_vol2[['closest_genus',
                                      'closest_epithet',
                                      'closest_infra_name',
                                      'authors',
                                      'taxon_rank',
                                      'taxon_rank_detailed']]
simplified_vol2.to_csv('../output/local/index_output/vol2_index_output.csv')
                                
simplified_vol3 = prune_authors_vol3[['closest_genus',
                                      'closest_epithet',
                                      'closest_infra_name',
                                      'authors',
                                      'taxon_rank',
                                      'taxon_rank_detailed']]
simplified_vol3.to_csv('../output/local/index_output/vol3_index_output.csv')

# %%
non_italics_simplified_vol1 = prune_authors_vol1.loc[(prune_authors_vol1['span_flags'] != 6),
                                                     ['closest_genus',
                                                      'closest_epithet',
                                                      'closest_infra_name',
                                                      'authors',
                                                      'taxon_rank',
                                                      'taxon_rank_detailed']]
non_italics_simplified_vol1.to_csv('../output/local/index_output/vol1_nonitalics.csv')

non_italics_simplified_vol2 = prune_authors_vol2.loc[(prune_authors_vol2['span_flags'] != 6),
                                                     ['closest_genus',
                                                      'closest_epithet',
                                                      'closest_infra_name',
                                                      'authors',
                                                      'taxon_rank',
                                                      'taxon_rank_detailed']]
non_italics_simplified_vol2.to_csv('../output/local/index_output/vol2_nonitalics.csv')

non_italics_simplified_vol3 = prune_authors_vol3.loc[(prune_authors_vol3['span_flags'] != 6),
                                                     ['closest_genus',
                                                      'closest_epithet',
                                                      'closest_infra_name',
                                                      'authors',
                                                      'taxon_rank',
                                                      'taxon_rank_detailed']]
non_italics_simplified_vol3.to_csv('../output/local/index_output/vol3_nonitalics.csv')

# %%


# %%



