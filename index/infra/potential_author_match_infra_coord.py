# Takes longer but makes more sense generally. We will skip it here
# def potential_author_match_infra_coord(row):
#     word = row['word']
#     pruned_word = row['pruned_word']
#     lower_word = word.lower()
#     latin_connectives = r"^\s?et[\s|.]?$|^\s?in[\s|.]?$|^\s?non[\s|.]?$|^\s?&[\s|.]?$|^\s?er[\s|.]?$|^\s?nec[\s|.]?$|^\s?mult[\s|.]?$|^\s?ex[\s|.]?$|^\s?fil[\s|.]?$"
#     infra_symbols = r"^var[\s|.|\b]?$|^subsp[\s|.|\b]?$|^ssp[\s|.|\b]?$|^spp[\s|.|\b]?$|^x[\s|.|\b]?$|^×[\s|.|\b]?$"
#     is_latin_connectives = re.search(latin_connectives, word) != None
#     is_infra_symbol = re.search(infra_symbols, lower_word) != None
#     if pruned_word:
#         is_upper_first = pruned_word[0].isupper()
#     else:
#         is_upper_first = False
#     return (not is_infra_symbol) and (is_upper_first or is_latin_connectives)

def potential_author_match_infra_coord(word):
    lower_word = word.lower()
    latin_connectives = r"^\s?et[\s|.]?$|^\s?in[\s|.]?$|^\s?non[\s|.]?$|^\s?&[\s|.]?$|^\s?er[\s|.]?$|^\s?nec[\s|.]?$|^\s?mult[\s|.]?$|^\s?ex[\s|.]?$|^\s?fil[\s|.]?$"
    infra_symbols = r"^var[\s|.|\b]?$|^subsp[\s|.|\b]?$|^ssp[\s|.|\b]?$|^spp[\s|.|\b]?$|^x[\s|.|\b]?$|^×[\s|.|\b]?$"
    is_latin_connectives = re.search(latin_connectives, word) != None
    is_infra_symbol = re.search(infra_symbols, lower_word) != None
    return (not is_infra_symbol) and (word[0].isupper() or is_latin_connectives)