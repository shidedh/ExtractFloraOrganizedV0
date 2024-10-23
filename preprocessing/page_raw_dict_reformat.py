def page_raw_dict_reformat(vol, page, page_num, word_list):
    """ 
    This function appends the characters and relevent information 
    corresponds to it in each page to word_list. Formatted such that 
    word_list is ready to represent each char as a pandas dataframe row 
    with column names:
        ['vol_num', 
         'page_num', 
         'block_num',
         'block_num_absolute', 
         'block_bbox',
         'line_num', 
         'line_wmode', 
         'line_dir', 
         'line_bbox', 
         'span_num',
         'span_size',
         'span_flags', 
         'span_font', 
         'span_color', 
         'span_ascender',
         'span_descender', 
         'span_origin', 
         'span_bbox', 
         'word_num', 
         'word',
         'char_num', 
         'char', 
         'char_origin', 
         'char_bbox']

    INPUTS: 
        vol: String corresponding to volume number 
        page: instance of document at index page_num
        page_num: Int corresponging to page number
        word_list: List to which each character will be appended
    """
    text_blocks = [
        block for block in page.get_text("rawdict")['blocks'] 
        if block['type'] == 0
        ]

    #for each block in text blocks
    for b_i in range(len(text_blocks)):
        b = text_blocks[b_i]
        curr_block_num_abs = b['number']                    #true blcok number
        curr_block_num_reletive = b_i                       #excludes image blocks
        curr_block_bbox = b['bbox']

        for l_i in range(len(b['lines'])):
            l = b['lines'][l_i]
            curr_line_num = l_i
            curr_line_wmode = l['wmode']
            curr_line_dir = l['dir']
            curr_line_bbox = l['bbox']
            
            word_num = 0
            for s_i in range(len(l['spans'])):
                s = l['spans'][s_i]
                span_num = s_i
                span_size = s['size']
                span_flags = s['flags']
                span_font = s['font']
                span_color = s['color']
                span_ascender = s['ascender']
                span_descender = s['descender'] 
                span_chars = s['chars'] 
                span_origin = s['origin'] 
                span_bbox = s['bbox']
                
                span_word = ""
                char_in_words = []
                
                for span_char_i in range(len(span_chars)):
                    c = span_chars[span_char_i]
                    char = c['c']

                    # if we are at the last character of span or a white space character
                    # add word to dictionary
                    if (span_char_i == len(span_chars) - 1) or \
                        (char.isspace() and len(span_word) > 0): 

                        if (span_char_i == len(span_chars) - 1) and (not char.isspace()): 
                            #to ensure the last character in the span is added (if it's not a space)
                            span_word += char
                            char_in_words.append(c)

                        for c_i in range(len(char_in_words)):
                            char_num = c_i 
                            char_origin = char_in_words[c_i]['origin']
                            char_bbox = char_in_words[c_i]['bbox']
                            curr_char = char_in_words[c_i]['c']

                            char_row = {
                                'vol_num': vol,
                                'page_num': page_num,
                                'block_num': curr_block_num_reletive,
                                'block_num_absolute': curr_block_num_abs,
                                'block_bbox': curr_block_bbox,

                                'line_num': curr_line_num,
                                'line_wmode': curr_line_wmode,
                                'line_dir': curr_line_dir,
                                'line_bbox': curr_line_bbox,
                                
                                'span_num': span_num,
                                'span_size': span_size,
                                'span_flags': span_flags,
                                'span_font': span_font,
                                'span_color': span_color,
                                'span_ascender': span_ascender,
                                'span_descender': span_descender,
                                'span_origin': span_origin,
                                'span_bbox': span_bbox,

                                'word_num': word_num, #in the entire line
                                'word': span_word,

                                'char_num': c_i,
                                'char': curr_char,
                                'char_origin': char_origin,
                                'char_bbox': char_bbox
                            }
                            word_list.append(char_row)

                        word_num += 1
                        span_word = ''
                        char_in_words = []
                    elif not char.isspace():  
                        span_word += char
                        char_in_words.append(c)
                    #only other possibility is that is it a white space in which case we can ignore it
