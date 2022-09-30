import os

def save_to_csv(df,fullfilename):
    """
    Create a folder with the filename in /kaggle/working directory
    Example:
    base_path_store = "/kaggle/working/data-stored"
    save_to_csv(df,f"{base_path_store}/savename.csv")
    """
    
    store_dirs = fullfilename.split("/")
    filename = store_dirs[-1]
    file_extension = filename.split('.')[-1]
    if '/'.join(store_dirs[:-2]) != '/kaggle/working':
        print('Error: the file should be stored under /kaggle/working.')
    elif file_extension != 'csv':
        print('Error: the extension of the file should be csv.')
    else:
        store_dir='/'.join(store_dirs[:-1])
        if not os.path.exists(store_dir):
            # Create a new directory because it does not exist 
            os.makedirs(store_dir)
            print(f"{store_dir} successfully created.")
        # save dataframe
        df.to_csv(fullfilename)
        print(f"{filename} successfully created.")
        

