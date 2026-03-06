from audioset_download import Downloader

d = Downloader(
    root_path='my_custom_test_data',

    # Point to your custom 20-line CSV file
    csv_path='my_test_file.csv', 

    # Set type to None, since we are using a custom CSV
    download_type=None, 

    n_jobs=4
)

d.download(format='wav')
