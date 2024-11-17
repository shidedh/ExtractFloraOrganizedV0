mkdir -p ./logs

search_dir="."

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="../logs/pipeline_${timestamp}.log"
packages="./logs/packages.log"


echo "Logging installed Python packages and their versions..." | tee "$packages"
pip freeze | tee -a "$packages"
grep -rE "^\s*(import|from)\s+" "$search_dir" | awk '{print $2}' | sort | uniq | tee -a "$packages"


# Measure the total time from start to end
{ time ( 
    cd ./preprocessing
    echo "Preprocessing is running..." | tee -a "$log_file"
    { time python3.10 get_char_df.py ; } 2>&1 | tee  -a "$log_file"
    echo "Preprocessing completed. Log file created at: $log_file" | tee -a "$log_file"
    cd ..

    cd ./index
    echo "Index is running..." | tee  -a "$log_file"
    { time python3.10 index.py ; } 2>&1 | tee  -a "$log_file"
    echo "Index completed. Log file created at: $log_file" | tee -a "$log_file"
    cd ..

    # do this for files in ./main_text
    cd ./main_text
    echo "Main text is running..." | tee -a "$log_file"
    echo "1) find_entry_boxes.py" | tee -a "$log_file"
    { time python3.10 find_entry_boxes.py ; } 2>&1 | tee -a "$log_file"
    echo "find_entry_boxes.py completed. Log file created at: $log_file" | tee -a "$log_file"
    echo "2) entry_bbox_page_cont.py" | tee -a "$log_file"
    { time python3.10 entry_bbox_page_cont.py ; } 2>&1 | tee -a "$log_file"
    echo "entry_bbox_page_cont.py completed. Log file created at: $log_file" | tee -a "$log_file"
    echo "3) parsing_locations.py" | tee -a "$log_file"
    { time python3.10 parsing_locations.py ; } 2>&1 | tee -a "$log_file"
    echo "parsing_locations.py completed. Log file created at: $log_file" | tee -a "$log_file"
    cd .. ) } 2>&1 | tee -a "$log_file"

# run chmod +x run_pipeline.sh to make this file executable 
# execute with ./run_pipeline.sh from the root directory




