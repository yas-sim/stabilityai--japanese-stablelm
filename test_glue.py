import datasets

raw_dataset = datasets.load_dataset('glue', 'mrpc', split='validation')

print(raw_dataset)
print(raw_dataset['idx'])

"""
Dataset({
    features: ['sentence1', 'sentence2', 'label', 'idx'],
    num_rows: 408
})
"""