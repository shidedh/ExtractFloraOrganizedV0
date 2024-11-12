# Reminder:
# all_vol_data_coord_match = [(vol1_char_df, vol1_index),
#                             (vol2_last8__char_df, vol2_last8__index),
#                             (vol3_char_df, vol3_index)]

from has_infra_symbols import has_infra_symbols
from potential_author_match_infra_coord import potential_author_match_infra_coord
all_vol_data_coord_match = ' YAML THING '
for vol_char_df, _ in all_vol_data_coord_match:
    vol_char_df["potential_infra_match"] = (vol_char_df['word'].apply(has_infra_symbols)) | \
                                           ((vol_char_df["infra_coord_match"] == True) & (vol_char_df['word'].apply(potential_author_match_infra_coord) == False))