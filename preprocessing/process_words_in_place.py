import numpy as np
# think about how vol_df should be fed into this function ...

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