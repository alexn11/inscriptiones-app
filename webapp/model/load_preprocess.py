
import pandas

from sklearn.preprocessing import MultiLabelBinarizer as skl_MultiLabelBinarizer

categories_list = [
  'tituli_honorarii',
  'tituli_operum',
  'tituli_sacri',
  'tituli_sepulcrales',
  'carmina',
  'defixiones',
  'diplomata_militaria',
  'miliaria',
  'inscriptiones_christianae',
  'leges',
  'senatus_consulta',
  'sigilla_impressa',
  'termini',
  'tesserae_nummulariae',
  'signacula',
  'signacula_medicorum',
  'litterae_erasae',
  'litterae_in_litura',
  'tituli_fabricationis',
  'tituli_possessionis',
  'tria_nomina',
  'viri',
  'Augusti_Augustae',
  'liberti_libertae',
  'milites',
  'mulieres',
  'nomen_singulare',
  'praenomen_et_nomen',
  'officium_professio',
  'ordo_decurionum',
  'ordo_equester',
  'ordo_senatorius',
  'reges',
  'sacerdotes_christiani',
  'sacerdotes_pagani',
  'servi_servae',
  'seviri_Augustales',
  'unknown_category',
]





def create_category_columns (inscriptions_data) :
  label_binarizer = skl_MultiLabelBinarizer (classes = categories_list)
  categories = inscriptions_data . genus_status . str . replace ('; +', ';', regex = True)
  categories = categories . str . replace (' +$', '', regex = True) . replace (' +', ' ', regex = True)
  categories = categories . str . replace ('[ /]', '_', regex = True)
  categories = categories . str . split (';')
  label_values = pandas . DataFrame (label_binarizer . fit_transform (categories), columns = categories_list, index = inscriptions_data . index)
  inscriptions_data = pandas . concat ([ inscriptions_data, label_values ], axis = 1)
  inscriptions_data . drop (columns = [ 'genus_status', ], inplace = True)
  return inscriptions_data



def load_data (file_name,
               do_accept_reconstructions = True,
               do_accept_corrections = True,
               inscription_min_length =1,
               process_blanks ='remove',
               do_remove_arabic_numbers = True,
               do_remove_breaks = True,
               #do_remove_empty_results = True,
               do_remove_erased_content = True,
               do_remove_non_latin_tags = True,
               do_remove_repeated_spaces = True,
               do_remove_separators = True,
               do_remove_suspected_forgeries = True,
               do_create_category_columns = False,
               do_cleanup = True) :

  inscriptions_data = pandas . read_csv (file_name)
  
  suspected_forgery_indicators = inscriptions_data . publication . str . contains ('*', regex = False)
  if (do_remove_suspected_forgeries) :
    inscriptions_data . drop (index = inscriptions_data [suspected_forgery_indicators] . index, inplace = True)
  else :
    inscriptions_data ['suspected_forgery'] = suspected_forgery_indicators
 
  if (process_blanks == 'remove') :
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace ('\[[36]\]', '', regex = True)
  elif (process_blanks == 'expand') :
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace ('\[3\]', ' ', regex = True)
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace ('\[6\]', '   ', regex = True)
  elif (process_blanks == 'keep') :
    pass
  else :
    raise Exception ('process_blank optional argument should be either "remove", "expand" or "keep"')
    
  if (do_remove_arabic_numbers) :
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace ('[0-9]', '', regex = True)
  else :
    if (any (incriptions_data . inscription . str . contains ('[0-9]', regex = True))) :
      print ('Found arabic numerals in the inscriptions')
    
  if (do_accept_reconstructions) :
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace ('(', '', regex = False) . str . replace (')', '', regex = False)
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace ('[', '', regex = False) . str . replace (']', '', regex = False)
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace (r'[?!]', '', regex = True)

  if (do_accept_corrections) :
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace (r'<([a-zA-Z]+)=[a-zA-Z]+>', r'\1', regex = True)
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace (r'<([a-zA-Z]+)=>', r'\1', regex = True)
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace (r'<([a-zA-Z]+) += +[a-zA-Z]+>', r'\1', regex = True)
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace (r'<([a-zA-Z]+)>', r'\1', regex = True)
    
  if (do_remove_breaks) :
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace ('/', '', regex = False)
    
  if (do_remove_erased_content) :
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace (r'\{.*?\}', '', regex = True)
  
  if (do_remove_non_latin_tags) :
    # greek
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace ('"GR"', '', regex = False)
    # punic?
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace ('"PUN"', '', regex = False)
    
  if (do_remove_separators) :
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace ('|', '', regex = False)
    
  if (do_cleanup) :
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace ('$', '', regex = False)
    
  if (do_remove_repeated_spaces) :
    inscriptions_data ['inscription'] = inscriptions_data . inscription . str . replace ('  +', ' ', regex = True)

  if (inscription_min_length > 0) :
    short_inscription_indexes = inscriptions_data [inscriptions_data . inscription . str . len () <= inscription_min_length] . index
    inscriptions_data . drop (index = short_inscription_indexes, inplace = True)
    
  inscriptions_data . genus_status . fillna ('unknown category', inplace = True)
  if (do_create_category_columns) :
    inscriptions_data = create_category_columns (inscriptions_data)
  else :
    inscriptions_data . rename (columns = { 'genus_status' : 'categories' }, inplace = True)
  
  return inscriptions_data








