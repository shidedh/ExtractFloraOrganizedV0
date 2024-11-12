from tqdm import tqdm
from get_center_x0 import get_center_x0
from get_col_num import get_col_num

# some yaml config thing to set these variables? 
all_vol_data_col_num = [(vol1_char_df, vol1_index, vol1_doc),
                        (vol2_char_df, vol2_index, vol2_doc),
                        (vol3_char_df, vol3_index, vol3_doc)]

for vol_char_df ,vol_index, doc in all_vol_data_col_num: 
    for page_num in tqdm(vol_index):
        center_x0 = get_center_x0(vol_char_df, page_num)
        #find center based on x0 coordinate of each line
        vol_char_df['col_num'] = vol_char_df['line_bbox'].apply(lambda coords : get_col_num(coords, center_x0)) 