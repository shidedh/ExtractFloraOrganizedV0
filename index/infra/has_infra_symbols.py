def has_infra_symbols(word):
    infra_symbols = r"^var[\s|.|\b]?$|^subsp[\s|.|\b]?$|^ssp[\s|.|\b]?$|^spp[\s|.|\b]?$|^x[\s|.|\b]?$|^×[\s|.|\b]?$"
    return re.search(infra_symbols, word) != None