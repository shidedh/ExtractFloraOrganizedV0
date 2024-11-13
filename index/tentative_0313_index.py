# %%
import fitz
import numpy as np
import pandas as pd
from tqdm import tqdm

import io
from PIL import Image, ImageDraw, ImageFont, ImageColor

import math
import re

# %%
vol1_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 1.pdf'
vol2_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 2.pdf'
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
vol2_index = list(range(703, 725))
vol3_index = list(range(555, 583))

# %%
TARGET_DPI = 300
mat = fitz.Matrix(TARGET_DPI/ 72, TARGET_DPI/ 72)

# %% [markdown]
# ### finding the columns 
# ### & checking if a word is a strict match for the genus / epithet pattern

# %%
def epithet_match(row):
    return row['word_num'] == 0 and \
           row['word'].isalpha() and \
           row['word'].islower()

def genus_match(row):
    return row['word_num'] == 0 and \
           row['word'].isalpha() and \
           row['word'][0].isupper() and row['word'][1:].islower()

# %%
#rightmost point of any bounding box:
def get_center_x0(vol_char_df, page_num, bias = 30):
    """WARNING: large bias causes miscatagorization in page number in book"""
    df = vol_char_df[vol_char_df['page_num'] == page_num]
    
    right_bound = df['line_bbox'].apply(lambda x : x[2]).max() 
    #leftmost point of any bounding box:
    left_bound = df['line_bbox'].apply(lambda x : x[0]).min()

    return 0.5*(right_bound + left_bound) - bias


def get_col_num(coords, center_x0):
    x0, y0, x1, y1 = coords
    return int(x0 >= center_x0)


all_vol_data = [(vol1_char_df, vol1_index, vol1_doc),
                (vol2_char_df, vol2_index, vol2_doc),
                (vol3_char_df, vol3_index, vol3_doc)]

for vol_char_df ,vol_index, doc in all_vol_data: 
    #for each volume check if genus pattern / epithet pattern exists within the index part of the book
    vol_char_df['genus_index_pat_match'] = vol_char_df.apply(lambda r : r['page_num'] in vol_index and genus_match(r), axis = 1) #does this for whole books which is bad
    vol_char_df['epithet_index_pat_match'] = vol_char_df.apply(lambda r : r['page_num'] in vol_index and epithet_match(r), axis = 1) #does this for whole books which is bad
    
    for page_num in tqdm(vol_index):
        center_x0 = get_center_x0(vol_char_df, page_num)
        #find center based on x0 coordinate of each line
        vol_char_df['col_num'] = vol_char_df['line_bbox'].apply(lambda coords : get_col_num(coords, center_x0)) 

# %% [markdown]
# #### testing if col num correctly assigned

# %%
all_vol_data = [(vol1_char_df, vol1_index, vol1_doc, "index_col_vol1"),
                (vol2_char_df, vol2_index, vol2_doc, "index_col_vol2"),
                (vol3_char_df, vol3_index, vol3_doc, "index_col_vol3")]

for vol_char_df, vol_index, vol_doc, output_name in all_vol_data:
    image_list = []
    keep_cols = vol_char_df.columns.difference(["char_num", "char", "char_origin", "char_bbox", "char_x0", "char_y0", "char_x1", "char_y1", "pruned_char_x0", "pruned_char_y0", "pruned_char_x1", "pruned_char_y1"], sort=False).tolist()
    for page_num in tqdm(vol_index):
        pix_map = vol_doc.get_page_pixmap(page_num,matrix=mat)
        image = Image.open(io.BytesIO(pix_map.tobytes()))
        draw = ImageDraw.Draw(image)

        temp_df = vol_char_df[vol_char_df["page_num"] == page_num].loc[:, keep_cols].drop_duplicates()

        for coord in temp_df[temp_df['col_num'] == 0]['line_bbox'] :
            x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
            draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#FF7F50"), width=5)

        for coord in temp_df[temp_df['col_num'] == 1]['line_bbox']:
            x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
            draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#003399"), width=5)
            
        image_list.append(image)
        #save pages of the volume
    image_list[0].save('../output/local/'+output_name+'.pdf' ,save_all=True, append_images=image_list[1:])


# %% [markdown]
# ### Genus / epithet flagging 
# flagging pages where number of strict genus or epithet patern matches is less than 3 per column

# %%
all_vol_data = [(vol1_char_df, vol1_index, vol1_doc, "strickt_match_vol1"),
                (vol2_char_df, vol2_index, vol2_doc, "strickt_match_vol2"),
                (vol3_char_df, vol3_index, vol3_doc, "strickt_match_vol3")]

for vol_char_df, vol_index, doc, output_name in all_vol_data: 
    #for each volume 
    image_list = []
    genus_flag_list = []
    epithet_flag_list = []
    for page_num in tqdm(vol_index):
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
# ### match  based on coordinates

# %%
def is_coord_match(x, x_ref_left, x_ref_right, margin):
    return (x_ref_left - margin <= x[0] and x[0] <= x_ref_left + margin) or (x_ref_right - margin <= x[0] and x[0] <= x_ref_right + margin)

# %% [markdown]
# #### epithet

# %%
all_vol_data = [(vol1_char_df, vol1_index),
                (vol2_char_df, vol2_index),
                (vol3_char_df, vol3_index)]

for vol_char_df, vol_index in all_vol_data: 
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
all_vol_data = [(vol1_char_df, vol1_index, vol1_doc, "epithet_coord_match_pruned_vol1"),
                (vol2_char_df, vol2_index, vol2_doc, "epithet_coord_match_pruned_vol2"),
                (vol3_char_df, vol3_index, vol3_doc, "epithet_coord_match_pruned_vol3")]

for vol_char_df, vol_index, doc, output_name in all_vol_data: 
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

# %% [markdown]
# #### Genus coord match

# %%
# add something about genus should come before epithet? 
    # assert df[df['epithet_coord_match'] == True]['word_bbox'].apply(lambda x: x[0]).mean() 
    #     >  df[df['genus_coord_match'] == True]['word_bbox'].apply(lambda x: x[0]).mean() 
    # and if False it shouldn't be a genus_coord?

# %%
all_vol_data = [(vol1_char_df, vol1_index),
                (vol2_char_df, vol2_index),
                (vol3_char_df, vol3_index)]

for vol_char_df, vol_index in all_vol_data: 
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
all_vol_data = [(vol1_char_df, vol1_index, vol1_doc, "genus_coord_match_vol1"),
                (vol2_char_df, vol2_index, vol2_doc, "genus_coord_match_vol2"),
                (vol3_char_df, vol3_index, vol3_doc, "genus_coord_match_vol3")]

for vol_char_df, vol_index, doc, output_name in all_vol_data: 
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
# ### Add column for genus / epithet coord mean for each page

# %%
# all_vol_data = [(vol1_char_df, vol1_index),
#                 (vol2_char_df, vol2_index),
#                 (vol3_char_df, vol3_index)]
# for vol_char_df, vol_index in all_vol_data:
#     for page_num in vol_index:
#         for c_i in [0, 1]:
#             genus_mean_coord = vol_char_df[(vol_char_df['page_num'] == page_num) & (vol_char_df['genus_coord_match'] == True) & (vol_char_df['col_num'] == c_i)]['word_bbox'].apply(lambda x: x[0]).mean()
#             epithet_mean_coord = vol_char_df[(vol_char_df['page_num'] == page_num) & (vol_char_df['epithet_coord_match'] == True) & (vol_char_df['col_num'] == c_i)]['word_bbox'].apply(lambda x: x[0]).mean()
        
#             #doing this because you can have no genus in one page but not no genus but an epithet...
#             if np.isnan(genus_mean_coord):
#                 genus_mean_coord == 0
#             if np.isnan(epithet_mean_coord):
#                 epithet_mean_coord = 1

#             vol_char_df.loc[(vol_char_df['page_num'] == page_num) & (vol_char_df['col_num'] == c_i), 'genus_mean_coord'] = genus_mean_coord
#             vol_char_df.loc[(vol_char_df['page_num'] == page_num) & (vol_char_df['col_num'] == c_i), 'epithet_mean_coord'] = epithet_mean_coord

# %% [markdown]
# ### extract potential genus / epithet matches

# %%
def potential_genus_match(row):
    return row['genus_coord_match'] == True and \
           row['epithet_coord_match'] == False and \
           row['word'].isupper() == False and \
           row['word'].isnumeric() == False and \
           row['word'].find("Flore") == -1 
           # removing this for now ... and row['genus_mean_coord'] < row['epithet_mean_coord'] #important to check this only when epithet_coord_match is false?

def potential_epithet_match(row):
    return row['epithet_coord_match'] == True and \
           row['word'].isupper() == False and \
           row['word'].isnumeric() == False

# %% [markdown]
# 

# %%
vol1_char_df['potential_genus_match'] = vol1_char_df.apply(potential_genus_match, axis = 1)
vol1_char_df['potential_epithet_match'] = vol1_char_df.apply(potential_epithet_match, axis = 1)

vol2_char_df['potential_genus_match'] = vol2_char_df.apply(potential_genus_match, axis = 1)
vol2_char_df['potential_epithet_match'] = vol2_char_df.apply(potential_epithet_match, axis = 1)

vol3_char_df['potential_genus_match'] = vol3_char_df.apply(potential_genus_match, axis = 1)
vol3_char_df['potential_epithet_match'] = vol3_char_df.apply(potential_epithet_match, axis = 1)

# %%
all_vol_data = [(vol1_char_df, vol1_index, vol1_doc, "GE_potential_match_vol1"),
                (vol2_char_df, vol2_index, vol2_doc, "GE_potential_match_vol2"),
                (vol3_char_df, vol3_index, vol3_doc, "GE_potential_match_vol3")]

for vol_char_df, vol_index, doc, output_name in all_vol_data: 
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

# %%
# all_vol_data = [(vol1_char_df, vol1_index),
#                 (vol2_char_df, vol2_index),
#                 (vol3_char_df, vol3_index)]
# for vol_char_df, vol_index in all_vol_data:
#     for page_num in vol_index: 
#         for col_num in [0,1]:
#             if vol_char_df["genus_mean_coord"]
#                 print(page_num, col_num)

# %% [markdown]
# ### infra species

# %%
all_vol_data = [(vol1_char_df, vol1_index),
                (vol2_char_df, vol2_index),
                (vol3_char_df, vol3_index)]

for vol_char_df, vol_index in all_vol_data: 
    vol_char_df["infra_coord_match"] = vol_char_df["word_bbox"].apply(lambda x : False)
    for page_num in tqdm(vol_index):

        margin = 1.25 * vol_char_df[(vol_char_df["epithet_coord_match"] == True) | (vol_char_df["genus_coord_match"] == True)]["char_bbox"].apply(lambda x : x[2] - x[0]).mean()
        
        mean_left_epithet = vol_char_df[(vol_char_df["page_num"] == page_num) & (vol_char_df["col_num"] == 0) & (vol_char_df["epithet_coord_match"] == True)]["word_bbox"].apply(lambda x : x[0]).mean()
        mean_left_genus = vol_char_df[(vol_char_df["page_num"] == page_num) & (vol_char_df["col_num"] == 0) & (vol_char_df["genus_coord_match"] == True)]["word_bbox"].apply(lambda x : x[0]).mean()
        if math.isnan(mean_left_genus):
            mean_left_genus_all = vol_char_df[(vol_char_df["col_num"] == 0) & (vol_char_df["genus_coord_match"] == True)]["word_bbox"].apply(lambda x : x[0]).mean()
            mean_left_epithet_all = vol_char_df[(vol_char_df["col_num"] == 0) & (vol_char_df["epithet_coord_match"] == True)]["word_bbox"].apply(lambda x : x[0]).mean()
            mean_left_tab = mean_left_epithet_all - mean_left_genus_all
        else: 
            mean_left_tab = mean_left_epithet - mean_left_genus
        
        mean_right_epithet = vol_char_df[(vol_char_df["page_num"] == page_num) & (vol_char_df["col_num"] == 1) & (vol_char_df["epithet_coord_match"] == True)]["word_bbox"].apply(lambda x : x[0]).mean()
        mean_right_genus = vol_char_df[(vol_char_df["page_num"] == page_num) & (vol_char_df["col_num"] == 1) & (vol_char_df["genus_coord_match"] == True)]["word_bbox"].apply(lambda x : x[0]).mean()
        if math.isnan(mean_right_genus):
            mean_right_genus_all = vol_char_df[(vol_char_df["col_num"] == 1) & (vol_char_df["genus_coord_match"] == True)]["word_bbox"].apply(lambda x : x[0]).mean()
            mean_right_epithet_all = vol_char_df[(vol_char_df["col_num"] == 1) & (vol_char_df["epithet_coord_match"] == True)]["word_bbox"].apply(lambda x : x[0]).mean()
            mean_right_tab = mean_right_epithet_all - mean_right_genus_all
        else: 
            mean_right_tab = mean_right_epithet - mean_right_genus

        vol_char_df.loc[(vol_char_df["page_num"] == page_num) & (vol_char_df["word_num"] == 0)  , "infra_coord_match"] = vol_char_df[(vol_char_df["page_num"] == page_num) & (vol_char_df["word_num"] == 0)]["pruned_word_bbox"].apply(lambda x : is_coord_match(x, mean_left_epithet + mean_left_tab, mean_right_epithet + mean_right_tab, margin))

# %%
all_vol_data = [(vol1_char_df, vol1_index),
                (vol2_char_df, vol2_index),
                (vol3_char_df, vol3_index)]

for vol_char_df, vol_index in all_vol_data: 
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
def potential_author_match_infra_coord(word):
    lower_word = word.lower()
    latin_connectives = r"^\s?et[\s|.]?$|^\s?in[\s|.]?$|^\s?non[\s|.]?$|^\s?&[\s|.]?$|^\s?er[\s|.]?$|^\s?nec[\s|.]?$|^\s?mult[\s|.]?$|^\s?ex[\s|.]?$|^\s?fil[\s|.]?$"
    infra_symbols = r"^var[\s|.|\b]?$|^subsp[\s|.|\b]?$|^ssp[\s|.|\b]?$|^spp[\s|.|\b]?$|^x[\s|.|\b]?$|^×[\s|.|\b]?$"
    is_latin_connectives = re.search(latin_connectives, word) != None
    is_infra_symbol = re.search(infra_symbols, lower_word) != None
    return (not is_infra_symbol) and (word[0].isupper() or is_latin_connectives)

# %%
potential_author_match_infra_coord("fil.")

# %%
all_vol_data = [(vol1_char_df, vol1_index),
                (vol2_char_df, vol2_index),
                (vol3_char_df, vol3_index)]
for vol_char_df, _ in all_vol_data:
    vol_char_df["potential_infra_match"] = (vol_char_df["infra_coord_match"] == True) & (vol_char_df['word'].apply(potential_author_match_infra_coord) == False)

# %%
def has_infra_symbols(word):
    infra_symbols = r"^var[\s|.|\b]?$|^subsp[\s|.|\b]?$|^ssp[\s|.|\b]?$|^spp[\s|.|\b]?$|^x[\s|.|\b]?$|^×[\s|.|\b]?$"
    return re.search(infra_symbols, word) != None

# %%
all_vol_data = [(vol1_char_df, vol1_index, vol1_doc, "potential_infra_match_vol1"),
                (vol2_char_df, vol2_index, vol2_doc, "potential_infra_match_vol2"),
                (vol3_char_df, vol3_index, vol3_doc, "potential_infra_match_vol3")][::-1]

for vol_char_df, vol_index, doc, output_name in all_vol_data: 
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
# ### functions for author matching 
# to detect anamolies in epithet and infra indentations

# %%
vol1_char_df['index_page_num'] = vol1_char_df['page_num'] - vol1_index[0] + 1
vol2_char_df['index_page_num'] = vol2_char_df['page_num'] - vol2_index[0] + 1
vol3_char_df['index_page_num'] = vol3_char_df['page_num'] - vol3_index[0] + 1

# %%
vol1_char_df[(vol1_char_df['potential_infra_match'] == True) & (vol1_char_df['word'].apply(has_infra_symbols) == False)][["index_page_num", "word"]].drop_duplicates()

# %%
vol2_char_df[(vol2_char_df['potential_infra_match'] == True) & (vol2_char_df['word'].apply(has_infra_symbols) == False)][["index_page_num", "word"]].drop_duplicates()

# %%
vol3_char_df[(vol3_char_df['potential_infra_match'] == True) & (vol3_char_df['word'].apply(has_infra_symbols) == False)][["index_page_num", "word"]].drop_duplicates()

# %% [markdown]
# #### upper case beggining / latin words in epithet coordd

# %%
def potential_author_match_epithet_coord(word):
    latin_connectives = r"^\s?et[\s|.]?$|^\s?in[\s|.]?$|^\s?non[\s|.]?$|^\s?&[\s|.]?$|^\s?er[\s|.]?$|^\s?nec[\s|.]?$|^\s?mult[\s|.]?$|^\s?ex[\s|.]?$|^\s?fil[\s|.]?$|^\s?f[\s|.]?$"
    is_latin_connectives = re.search(latin_connectives, word) != None
    is_hybrid = word == "X"
    return is_latin_connectives or (word[0].isupper() and (not is_hybrid))

# %%
vol1_char_df[(vol1_char_df['potential_epithet_match'] == True) & (vol1_char_df['word'].apply(potential_author_match_epithet_coord))][["index_page_num", "word"]].drop_duplicates()

# %%
vol2_char_df[(vol2_char_df['potential_epithet_match'] == True) & (vol2_char_df['word'].apply(potential_author_match_epithet_coord))][["index_page_num", "word"]].drop_duplicates()

# %%
vol3_char_df[(vol3_char_df['potential_epithet_match'] == True) & (vol3_char_df['word'].apply(potential_author_match_epithet_coord))][["index_page_num", "word"]].drop_duplicates()

# %% [markdown]
# #### epithet coord word has uppper case in the middle (but not the first letter)

# %%
def has_upper_not_first(word):
    return word[1:].lower() != word[1:]

# %%
vol1_char_df[(vol1_char_df['potential_epithet_match'] == True) & (vol1_char_df['word'].apply(has_upper_not_first))][["index_page_num", "word"]].drop_duplicates()

# %%
#vol1_char_df[(vol1_char_df['potential_epithet_match'] == True) & (vol1_char_df['word'].apply(has_upper_not_first))][["index_page_num", "word"]].drop_duplicates()
vol1_char_df.loc[(vol1_char_df['potential_epithet_match'] == True) & (vol1_char_df['word'].apply(has_upper_not_first)) & (vol1_char_df['word'].isin(['J.d,IlLIlU.'])), 'potential_genus_match'] = True
vol1_char_df.loc[(vol1_char_df['potential_epithet_match'] == True) & (vol1_char_df['word'].apply(has_upper_not_first)) & (vol1_char_df['word'].isin(['J.d,IlLIlU.'])), 'potential_epithet_match'] = False
#vol1_char_df[(vol1_char_df['potential_epithet_match'] == True) & (vol1_char_df['word'].apply(has_upper_not_first))][["index_page_num", "word"]].drop_duplicates()

# %%
# J.d,IlLIlU. -> unidentified genus 


# %%
vol2_char_df[(vol2_char_df['potential_epithet_match'] == True) & (vol2_char_df['word'].apply(has_upper_not_first))][["index_page_num", "word"]].drop_duplicates()

# %%
vol3_char_df[(vol3_char_df['potential_epithet_match'] == True) & (vol3_char_df['word'].apply(has_upper_not_first))][["index_page_num", "word"]].drop_duplicates()

# %%
not_epithet_list = ['Schiman-Czeika']
vol3_char_df.loc[(vol3_char_df['potential_epithet_match'] == True) & (vol3_char_df['word'].apply(has_upper_not_first)) & (vol3_char_df['word'].isin(not_epithet_list)), 'potential_epithet_match'] = False

# %%
vol3_char_df[(vol3_char_df['potential_epithet_match'] == True) & (vol3_char_df['word'].apply(has_upper_not_first))][["index_page_num", "word"]].drop_duplicates()

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
vol1_char_df = vol1_char_df.loc[~((vol1_char_df['potential_genus_match'] == True) & (vol1_char_df['word'].apply(flag_genus_name))), :]

# %%
vol2_char_df[(vol2_char_df['potential_genus_match'] == True) & (vol2_char_df['word'].apply(flag_genus_name))][["index_page_num", "word"]].drop_duplicates()
#VV.1l.* only removing 

# %%
vol2_char_df = vol2_char_df.loc[(vol2_char_df['word'] != 'VV.1l.*'),:]

# %%
vol3_char_df[(vol3_char_df['potential_genus_match'] == True) & (vol3_char_df['word'].apply(flag_genus_name))][["index_page_num", "word"]].drop_duplicates()

# %% [markdown]
# flag if we had 2 genus in the same line or 1 or more genus + 1 or more epithet on the same line

# %%
#doesn't pick up all the issues because sometimes when the space if large enough 
# it thinks we're on a "new line"

# %%
line_groups = [c for c in vol1_char_df.columns if c.startswith("vol")] + \
              [c for c in vol1_char_df.columns if c.startswith("page")] + \
              [c for c in vol1_char_df.columns if c.startswith("block")] +\
              [c for c in vol1_char_df.columns if c.startswith("line")]
              
line_group_df = vol1_char_df.groupby(line_groups)
temp_line_df = vol1_char_df[line_group_df['potential_genus_match'].transform('any') & line_group_df['potential_epithet_match'].transform('any')]
temp_line_df[(temp_line_df['potential_genus_match'] == True) | (temp_line_df['potential_epithet_match'] == True)][["page_num", "block_num", "line_num", "word"]].drop_duplicates()

# %%
line_groups = [c for c in vol2_char_df.columns if c.startswith("vol")] + \
              [c for c in vol2_char_df.columns if c.startswith("page")] + \
              [c for c in vol2_char_df.columns if c.startswith("block")] +\
              [c for c in vol2_char_df.columns if c.startswith("line")]
              
line_group_df = vol2_char_df.groupby(line_groups)
temp_line_df = vol2_char_df[line_group_df['potential_genus_match'].transform('any') & line_group_df['potential_epithet_match'].transform('any')]
temp_line_df[(temp_line_df['potential_genus_match'] == True) | (temp_line_df['potential_epithet_match'] == True)][["page_num", "block_num", "line_num", "word"]].drop_duplicates()

# %%
line_groups = [c for c in vol3_char_df.columns if c.startswith("vol")] + \
              [c for c in vol3_char_df.columns if c.startswith("page")] + \
              [c for c in vol3_char_df.columns if c.startswith("block")] +\
              [c for c in vol3_char_df.columns if c.startswith("line")]
              
line_group_df = vol3_char_df.groupby(line_groups)
temp_line_df = vol3_char_df[line_group_df['potential_genus_match'].transform('any') & line_group_df['potential_epithet_match'].transform('any')]
temp_line_df[(temp_line_df['potential_genus_match'] == True) | (temp_line_df['potential_epithet_match'] == True)][["page_num", "block_num", "line_num", "word"]].drop_duplicates()

# %% [markdown]
# ### Matching page number

# %%
all_vol_data = [(vol1_char_df, vol1_index, vol1_doc),
                (vol2_char_df, vol2_index, vol2_doc),
                (vol3_char_df, vol3_index, vol3_doc)]

for vol_char_df ,vol_index, doc in all_vol_data: 
    #for each volume check if genus pattern / epithet pattern exists within the index part of the book
    for page_num in tqdm(vol_index):
        center_x0 = get_center_x0(vol_char_df, page_num, - 30)
        #find center based on x0 coordinate of each line
        vol_char_df['col_num_for_PN'] = vol_char_df['line_bbox'].apply(lambda coords : get_col_num(coords, center_x0)) 

# %%
def is_page_num(row):
    return row['pruned_word'].isnumeric()


all_vol_data = [(vol1_char_df, vol1_index),
                (vol2_char_df, vol2_index),
                (vol3_char_df, vol3_index)]

for vol_char_df, vol_index in all_vol_data: 
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
all_vol_data = [(vol1_char_df, vol1_index, vol1_doc, "potential_page_num_match_vol1"),
                (vol2_char_df, vol2_index, vol2_doc, "potential_page_num_match_vol2"),
                (vol3_char_df, vol3_index, vol3_doc, "potential_page_num_match_vol3")][::-1]

for vol_char_df, vol_index, doc, output_name in all_vol_data: 
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
# ### testing highlighting instead of making image:

# %% [markdown]
# 

# %% [markdown]
# page.add_highlight_annot(quads)
# 

# %%
#vol3_char_df[''].apply()

# %% [markdown]
# ### marking all values in the dataframe

# %% [markdown]
# ### index df 

# %%
#making sure page_num is in index
#making sure the genus level word is not all uppercase (a family name)
#making sure the pruned_word is not numeric (removing page_number as it's not in order usually)


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
                                (~(vol_char_df["page_num_coord_match"] == True))
                                ].copy()
    result.append(curr_result_df)

vol1_index_df, vol2_index_df, vol3_index_df = result[0], result[1], result[2]

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
    vol_index_df['closest_infra_type'] = vol_index_df.apply(extract_potential_infra_type, axis = 1)
    vol_index_df.loc[(vol_index_df['potential_epithet_match'] == True) | (vol_index_df['potential_genus_match'] == True), 'closest_infra_type'] = -1
    vol_index_df['closest_infra_type'].ffill(inplace=True)

# %%
keep_cols = vol3_index_df.columns.difference(["char_num", "char", "char_origin", "char_bbox"], sort=False).tolist()

vol3_index_test = vol3_index_df.copy().loc[:,keep_cols].drop_duplicates().reset_index()
vol3_index_test.rename(columns={"index": "char_index"}, inplace = True)

# %%
for vol_index_df in [vol3_index_test]:#[vol1_index_df, vol2_index_df, vol3_index_df]:
    infra_name_match_indecies = vol_index_df[vol_index_df['potential_infra_match'] == True].index + 1
    vol_index_df['closest_infra_name'] = np.NaN
    vol_index_df.loc[infra_name_match_indecies, 'closest_infra_name'] = vol_index_df.apply(lambda row : row['word'] + "_" + str(row['page_num']) + "_" + str(row['block_num']) + "_" + str(row['line_num']) , axis = 1)
    vol_index_df['potential_epithet_name_match'] = vol_index_df.index.isin(infra_name_match_indecies)
    vol_index_df.loc[(vol_index_df['potential_epithet_match'] == True) | (vol_index_df['potential_genus_match'] == True), 'closest_infra_name'] = -1
    vol_index_df['closest_infra_name'].ffill(inplace=True)

# %%
vol3_index_test.replace(-1, np.NaN, inplace = True)

# %%
vol3_index_test.iloc[:,17:].head(50)

# %%
[c for c in vol3_index_test.columns if c.startswith('potential')]

# %%
vol3_index_test['potential_author_match'] = (vol3_index_test['potential_genus_match'] == False) & \
                                            (vol3_index_test['potential_epithet_match'] == False) & \
                                            (vol3_index_test['potential_infra_match'] == False) & \
                                            (vol3_index_test['potential_epithet_name_match'] == False)

# %%
#vol3_index_test[vol3_index_test['potential_infra_match'] == True].index + 1

# %%
#vol3_index_test.iloc[:,18:].head(50)
#genus author: genus = "genus" & "closest_epithet = -1 & potential_genus_match = False"
#epithet author: epithet = "epithet" & potential_epithet_match = False & closest_infra = -1 
#closest_infra 

# %%
#vol3_index_test = vol3_index_df.copy()
#vol3_index_test['after_potential_infra_match'] = vol3_index_test['potential_infra_match'].shift()

# group_cols = vol3_index_test.columns.difference(["char_num", "char", "char_origin", "char_bbox"], sort=False).tolist()
# vol3_index_test["after_potential_infra_match"] = vol3_index_test.groupby(group_cols)['potential_infra_match'].shift()#.transform('min')

# %%
# vol3_index_df[['word_num','word','word_bbox','pruned_word', 'pruned_word_bbox', 'potential_genus_match', 'potential_epithet_match', 'potential_infra_match']].drop_duplicates()

# %%
# vol3_index_test[['word_num','word','word_bbox','pruned_word', 'pruned_word_bbox', 'potential_genus_match', 'potential_epithet_match', 'potential_infra_match', 'after_potential_infra_match']].drop_duplicates().head(50)

# %%
# vol3_index_test.loc[vol3_index_test['after_potential_infra_match'] == True, ['word_num','word','word_bbox','pruned_word', 'pruned_word_bbox', 'potential_genus_match', 'potential_epithet_match', 'potential_infra_match', 'after_potential_infra_match']].drop_duplicates()

# %%
# #df['closest_epithet_v2'] = np.nan
# def extract_potential_infra_names(row):
#     if row['after_potential_infra_match'] == True:
#         return row['word']
#     else:
#         return np.nan

# for vol_index_df in [vol1_index_df, vol2_index_df, vol3_index_df]:
#     vol_index_df['closest_epithet'] = vol_index_df.apply(extract_potential_epithet_names, axis = 1)
#     df.loc[df['potential_genus_match'] == True, 'closest_epithet_v2'] = -1
#     vol_index_df['closest_epithet'].ffill(inplace=True)

# %%
# df.loc[:,['word_num','word','word_bbox','pruned_word', 'pruned_word_bbox', 'potential_genus_match', 'potential_epithet_match', 'potential_infra_match', 'closest_genus', 'closest_epithet']]#.drop_duplicates().tail(50)

# %%
# type(df.at[i, 'closest_genus'])

# %%
# closes_genus = df.at[i, 'closest_genus']
# pd.isnull(closes_genus) == False

# %%
# df.loc[df['potential_genus_match'] == True, 'closest_epithet'] = np.nan

# %%
# df.loc[:,['word_num','word','word_bbox','pruned_word', 'pruned_word_bbox', 'potential_genus_match', 'potential_epithet_match', 'potential_infra_match', 'closest_genus', 'closest_epithet']].drop_duplicates().tail(50)

# %%
# all_vol_data = [(vol1_char_df, vol1_index, vol1_doc, "potential_infra_match_vol1"),
#                 (vol2_char_df, vol2_index, vol2_doc, "potential_infra_match_vol2"),
#                 (vol3_char_df, vol3_index, vol3_doc, "potential_infra_match_vol3")][::-1]

# for vol_char_df, vol_index, doc, output_name in all_vol_data: 
#     #for each volume 
#     image_list = []

#     for page_num in tqdm(vol_index):
#         pix_map = doc.get_page_pixmap(page_num,matrix=mat)
#         image = Image.open(io.BytesIO(pix_map.tobytes()))
#         draw = ImageDraw.Draw(image)
        

#         infra_coord_db = vol_char_df[(vol_char_df['page_num'] == page_num) & 
#                                      (vol_char_df['infra_coord_match'] == True)
#                             ].loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin",	"char_bbox"])
#                             ].drop_duplicates()

#         infra_db = vol_char_df[(vol_char_df['page_num'] == page_num) 
#                                 & (vol_char_df['potential_infra_match'] == True)
#                                 ].loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin",	"char_bbox"])
#                                 ].drop_duplicates()

#         with_infra_symbols = vol_char_df[(vol_char_df['page_num'] == page_num) &
#                                          (vol_char_df['infra_coord_match'] == True) & 
#                                          (vol_char_df['word'].apply(has_infra_symbols) == True)
#                                         ].loc[:,~vol_char_df.columns.isin(["char_num", "char", "char_origin",	"char_bbox"])
#                                         ].drop_duplicates()

#         #genus Coord is orange-pinkish, 5
#         for coord in infra_coord_db['word_bbox'] :
#             x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
#             draw.rectangle((x0-5, y0-5, x1+5, y1+5), fill=None, outline=ImageColor.getrgb("#003399"), width=7)

#         for coord in infra_db['word_bbox'] :
#             x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
#             draw.rectangle((x0-3, y0-3, x1+3, y1+3), fill=None, outline=ImageColor.getrgb("#FF7F50"), width=5)
            
#         # #epithet is red, 3
#         for coord in with_infra_symbols['word_bbox'] :
#             x0, y0, x1, y1 = [f*TARGET_DPI/ 72 for f in coord]
#             draw.rectangle((x0, y0, x1, y1), fill=None, outline=ImageColor.getrgb("#990000"), width=3)

#         image_list.append(image)

#     #save pages of the volume
#     image_list[0].save('../output/local/'+output_name+'.pdf' ,save_all=True, append_images=image_list[1:])

# %%
vol3_index_test

# %%
vol3_index_test.replace(np.NaN, "",inplace = True)

# %%
author_grouping = ['closest_genus', 'closest_epithet', 'closest_infra_name']
vol3_index_test['potential_author_match']
# group by 'A' and 'B' columns
groups = vol3_index_test.groupby(author_grouping)

# concatenate 'D' values for each group where 'C' is False
def concatenate(group):
    return group.loc[group['potential_author_match'] == True, 'word'].str.cat(sep=' ')

concatenated = groups.apply(concatenate).reset_index()

# add the concatenated values to the original dataframe
result = vol3_index_test.merge(concatenated[['closest_genus', 'closest_epithet', 'closest_infra_name', 0]], on=['closest_genus', 'closest_epithet', 'closest_infra_name'], how='left').rename(columns={0: 'authors'})

# %%
result.iloc[:,20:]

# %%
def fix_words(word):
    head, sep, tail = word.partition('_')
    return head 

result['closest_genus'] = result['closest_genus'].apply(fix_words)
result['closest_epithet'] = result['closest_epithet'].apply(fix_words)
result['closest_infra_type'] = result['closest_infra_type'].apply(fix_words)
result['closest_infra_name'] = result['closest_infra_name'].apply(fix_words)

# %%
result_prune_authors = result[(result['potential_genus_match'] == True) |
                              (result['potential_epithet_match'] == True) |
                              (result['potential_epithet_name_match'] == True)]

# %%
[c for c in vol3_index_test.columns if c.startswith('closest')]

# %%


# %%
simplified_result = result_prune_authors[['closest_genus',
                                          'closest_epithet',
                                          'closest_infra_type',
                                          'closest_infra_name',
                                          'authors']]

# %%
simplified_result.to_csv('vol3_index_output_v2.csv')

# %%
non_italics_simplified_result = result_prune_authors.loc[(result_prune_authors['span_flags'] != 6),
                                                     ['closest_genus',
                                                      'closest_epithet',
                                                      'closest_infra_type',
                                                      'closest_infra_name',
                                                      'authors']]

non_italics_simplified_result.to_csv('vol3_nonitalics_index_output_v2.csv')

# %%
text = 'closest_infra_name'
head, sep, tail = text.partition('_')

# %%
result_prune_authors.columns

# %%


# %%
simplified_result[((simplified_result['closest_genus'].str.contains('x')) | (simplified_result['closest_genus'].str.contains('x')) | (simplified_result['closest_genus'].str.contains('×'))) &
                  (simplified_result['closest_genus'].apply(lambda x : len(x)) <= 3)]

# %%
simplified_result[((simplified_result['closest_epithet'].str.contains('x')) | (simplified_result['closest_epithet'].str.contains('x')) | (simplified_result['closest_epithet'].str.contains('×'))) &
                  (simplified_result['closest_epithet'].apply(lambda x : len(x)) <= 3)]

# %%
simplified_result[((simplified_result['closest_infra_type'].str.contains('x')) | (simplified_result['closest_infra_type'].str.contains('x')) | (simplified_result['closest_infra_type'].str.contains('×'))) &
                  (simplified_result['closest_infra_type'].apply(lambda x : len(x)) <= 3)]

# %%
simplified_result[((simplified_result['closest_infra_name'].str.contains('x')) | (simplified_result['closest_infra_name'].str.contains('x')) | (simplified_result['closest_infra_name'].str.contains('×')))]

# %%
simplified_result[~((simplified_result['authors'].str.contains('ex')) | (simplified_result['authors'].str.contains('ex')) | (simplified_result['authors'].str.contains('e×'))) &
    ((simplified_result['authors'].str.contains('x')) | (simplified_result['authors'].str.contains('x')) | (simplified_result['authors'].str.contains('×')))]

# %%
# author_split = simplified_result[~((simplified_result['authors'].str.contains('ex')) | (simplified_result['authors'].str.contains('ex')) | (simplified_result['authors'].str.contains('e×'))) &
#     ((simplified_result['authors'].str.contains('x')) | (simplified_result['authors'].str.contains('x')) | (simplified_result['authors'].str.contains('×')))]#['authors'].str.split()#.apply(lambda x : x[0] == 'x')

vol3_char_df[(vol3_char_df['word'] == 'x') & (vol3_char_df['potential_genus_match'] == True)]

# %%
vol3_char_df['is_hybrid'] = False

# %%
index_end = len(vol3_char_df.loc[1584638+1,'word'])
vol3_char_df.loc[1584638,'potential_genus_match'] = False
vol3_char_df.loc[1584638+1:1584638+index_end,'potential_genus_match'] = True
vol3_char_df.loc[1584638+1:1584638+index_end,'potential_epithet_match'] = False
vol3_char_df.loc[1584638+1:1584638+index_end,'is_hybrid'] = True

# %%
vol3_char_df[(vol3_char_df['word'] == 'x') & (vol3_char_df['potential_epithet_match'] == True)]

# %%
author_split = simplified_result[~((simplified_result['authors'].str.contains('ex')) | (simplified_result['authors'].str.contains('ex')) | (simplified_result['authors'].str.contains('e×'))) &
    ((simplified_result['authors'].str.contains('x')) | (simplified_result['authors'].str.contains('x')) | (simplified_result['authors'].str.contains('×')))]#['authors'].str.split()#.apply(lambda x : x[0] == 'x')

# %%
author_split[author_split['authors'].str.split().apply(lambda x : x[0] == 'x')]

# %%
author_split = author_split[author_split['authors'].str.split().apply(lambda x : x[0] == 'x')]

# %%
author_split['closest_infra_type'] = 'hybrid'
author_split['closest_infra_name'] = author_split['authors'].apply(lambda x : x[1:])
author_split['authors'] = author_split['authors'].apply(lambda x : x[2:])

# %%
author_split

# %%
author_split['authors'].apply(lambda x : x.split()[1:])

# %%


# %%



