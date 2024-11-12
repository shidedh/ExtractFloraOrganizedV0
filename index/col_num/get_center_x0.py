#rightmost point of any bounding box:
def get_center_x0(vol_char_df, page_num, bias = 30):
    """WARNING: Bias = 30 large bias causes miscatagorization in page number in book"""
    df = vol_char_df[vol_char_df['page_num'] == page_num]
    
    right_bound = df['line_bbox'].apply(lambda x : x[2]).max() 
    #leftmost point of any bounding box:
    left_bound = df['line_bbox'].apply(lambda x : x[0]).min()

    return 0.5*(right_bound + left_bound) - bias