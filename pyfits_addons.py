from astropy.io import fits

def add_new_columns(input_file, output_file, hdu_index, new_column_name, new_column_data, new_column_format):
    """
    Add new columns to fits file.
    
    Parameters:
    input_file (str): Path to the input FITS file.
    output_file (str): Path to the output FITS file.
    hdu_index (int): Index of the HDU to add the new column to.
    new_column_name (str): The name of the new column to add.
    new_column_data (array-like): The data for the new column to add.
    new_column_format (str): The format of the new column to add.
    """

    with fits.open(input_file) as hdul:
        print(hdul.info())
        for i, hdu in enumerate(hdul):
            if i == hdu_index:
                if isinstance(hdu, fits.BinTableHDU):
                    columns = hdu.columns
                    data = hdu.data
                    if len(new_column_data) != len(data):
                        print('The length of new_column_data is not equal to the length of the existing data.')
                        break
                    columns.add_col(fits.Column(name=new_column_name, format=new_column_format, array=new_column_data))
                    print('-'*50)
                    print('new column added')
                    print(f'hdu : {hdu.name}')
                    break 
                else:
                    print('This is not BinTableHDU')
                    print('Please check the hdu_index!')
                    break

        hdul_new = fits.HDUList(hdul)
        hdul_new.writeto(output_file, overwrite=True)

def del_columns(input_file, output_file, hdu_index, del_column_name):
    """
    Add new columns to fits file.
    
    Parameters:
    input_file (str): Path to the input FITS file.
    output_file (str): Path to the output FITS file.
    hdu_index (int): Index of the HDU to add the new column to.
    new_column_name (str): The name of the new column to add.
    """

    with fits.open(input_file) as hdul:
        print(hdul.info())
        for i, hdu in enumerate(hdul):
            if i == hdu_index:
                if isinstance(hdu, fits.BinTableHDU):
                    columns = hdu.columns
                    data = hdu.data
                    if del_column_name not in columns.names:
                        print('The column name does not exist in the existing columns.')
                        break
                    columns.del_col(del_column_name)
                    print('-'*50)
                    print('column deleted')
                    print(f'hdu : {hdu.name}')
                    break 
                else:
                    print('This is not BinTableHDU')
                    print('Please check the hdu_index!')
                    break

        hdul_new = fits.HDUList(hdul)
        hdul_new.writeto(output_file, overwrite=True)