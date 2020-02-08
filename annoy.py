from annoy import AnnoyIndex

def build_annoy_index(encoding_dim, num_trees, annoy_index_file, encodings):
  ann = AnnoyIndex(encoding_dim)
  for index, encoding in enumerate(encodings):
    ann.add_item(index, encoding)
  ann.build(num_trees)
  ann.save(annoy_index_file)
  print("Created Annoy Index Successfully")