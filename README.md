# ExtractFloraOrganizedV0
First draft of reorganization of the extract flora pipeline

Purpose of this version / branch: 
I am putting together a first draft for Maryam's dissertation in this branch. Since I tried many different things to more comprehensively cover the nuances of the flora, I have multiple scripts that are almost identical. So in the interest of time I am only putting together a working version. We will work on a better organized and complete version. Logically this version should give us a lower bound for the accuracy of the output. 

Important dependencies / prereqs: 
- PyMuPDF 1.21.1: Python bindings for the MuPDF 1.21.1 library.
Version date: 2022-12-13 00:00:01.
Built for Python 3.10 on darwin (64-bit).
- please make the output/local/index_output , output/Oct2024 folder ,and output/local/main_text/ folders in root repo before running index script (will build the right repos in the script in future versions)
- pdfs: 
    - NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 1.pdf
    - NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 2.pdf
    - NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 3.pdf
    - vol2_last_8_pages.pdf
- pandas, PyMuPDF (fitz) -- version 1.21.1, matplotlib, tqdm, etc. 

Order of running scripts:
- prereqs (above section)
- preprocessing
    - get_char_df.py
- index
    - index.py
- main_text
    - find_entry_boxes.py
    - entry_bbox_page_cont.py
    - parsing_locations.py

    