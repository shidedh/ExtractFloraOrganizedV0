### YAML CONFIG FOR DRAW ### 


all_vol_dat_infra_match_test = [(vol1_char_df, vol1_index, vol1_doc, "potential_infra_match_vol1"),
                                (vol2_last8__char_df, vol2_last8__index, vol2_last8__doc, "potential_infra_match_vol2_last8_"),
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