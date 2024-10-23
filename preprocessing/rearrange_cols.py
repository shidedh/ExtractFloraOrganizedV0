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
    