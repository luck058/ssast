from audioset_download import Downloader

d = Downloader(
    root_path='my_test_data',
    
    # 1. Ask for only one label
    labels=['Speech'], 
    
    # 2. Use the smallest dataset (eval)
    download_type='eval', 
    
    n_jobs=4
)

print("Starting download of small test set...")
d.download(format='wav')
print("Test set download complete.")
