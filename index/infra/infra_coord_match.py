# Reminder:
# all_vol_data_coord_match = [(vol1_char_df, vol1_index),
#                             (vol2_last8__char_df, vol2_last8__index),
#                             (vol3_char_df, vol3_index)]
# YAML CONFIG

from tqdm import tqdm
from find_genus_epithet.coord_match.is_coord_match import is_coord_match # need to fix this with an __init__ method
import math

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