import re

class TextCleaner(object):
    def __init__(self):
        pass
        
    def remove_special_characters(self, paragraphs):
        paragraphs = [re.sub('[!,*)@#%(&$_?^]', '', paragraph) for paragraph in paragraphs]

        return paragraphs

    def remove_start_end_spaces(self, paragraphs):
        paragraphs = [paragraph for paragraph in paragraphs if paragraph.strip()]

        return paragraphs

    def remove_multiple_spaces(self, paragraphs):
        paragraphs = [' '.join(paragraph.split()) for paragraph in paragraphs] 

        return paragraphs

    def remove_start_numbers(self, paragraphs):
        paragraphs = [paragraph.lstrip('0123456789.- ') for paragraph in paragraphs if paragraph.strip()]

        return paragraphs

    def clean(self, paragraphs):
        paragraphs = self.remove_special_characters(paragraphs)
        paragraphs = self.remove_start_end_spaces(paragraphs)
        paragraphs = self.remove_multiple_spaces(paragraphs)
        paragraphs = self.remove_start_numbers(paragraphs)

        return paragraphs

# def remove_special_characters(paragraphs):
#     paragraphs = [re.sub('[!,*)@#%(&$_?^]', '', paragraph) for paragraph in paragraphs]
    
#     return paragraphs

# def remove_start_end_spaces(paragraphs):
#     paragraphs = [paragraph for paragraph in paragraphs if paragraph.strip()]
    
#     return paragraphs

# def remove_multiple_spaces(paragraphs):
#     paragraphs = [' '.join(paragraph.split()) for paragraph in paragraphs] 
    
#     return paragraphs

# def remove_start_numbers(paragraphs):
#     paragraphs = [paragraph.lstrip('0123456789.- ') for paragraph in paragraphs if paragraph.strip()]
    
#     return paragraphs

# def clean(paragraphs):
#     paragraphs = remove_special_characters(paragraphs)
#     paragraphs = remove_start_end_spaces(paragraphs)
#     paragraphs = remove_multiple_spaces(paragraphs)
#     paragraphs = remove_start_numbers(paragraphs)
    
#     return paragraphs