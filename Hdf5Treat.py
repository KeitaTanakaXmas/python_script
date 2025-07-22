import h5py
import pandas as pd

class Hdf5Command:

    def __init__(self,savehdf5):
        self.savehdf5 = savehdf5
        self.data_list = {}

    def descend_obj(self,obj,sep='\t'):
        """
        Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
        """
        if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
            for key in obj.keys():
                print (sep,'-',key,':',obj[key])
                self.descend_obj(obj[key],sep=sep+'\t')
        elif type(obj)==h5py._hl.dataset.Dataset:
            for key in obj.attrs.keys():
                print (sep+'\t','-',key,':',obj.attrs[key])

    def descend_obj_data(self,obj,sep='\t'):
        """
        Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
        """
        if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
            for key in obj.keys():
                print (sep,'-',key,':',obj[key])
                if isinstance(obj[key], h5py.Dataset):
                    name = obj[key].name
                    self.data_list[name] = obj[key][...]
                self.descend_obj_data(obj[key],sep=sep+'\t')
        elif type(obj)==h5py._hl.dataset.Dataset:
            for key in obj.attrs.keys():
                print(key)
                print (sep+'\t','-',key,':',obj.attrs[key])

    def dump(self,group='/'):
        """
        print HDF5 file metadata

        group: you can give a specific group, defaults to the root group
        """
        with h5py.File(self.savehdf5,'r') as f:
             self.descend_obj(f[group])

    def dump_data(self,group='/'):
        """
        print HDF5 file metadata

        group: you can give a specific group, defaults to the root group
        """
        self.data_list = {}
        with h5py.File(self.savehdf5,'r') as f:
             self.descend_obj_data(f[group])
        print(self.data_list)
        return self.data_list
        
    def export_csv(self,data,savecsv):
        df = pd.DataFrame(data)
        print(df)
        df.to_csv(savecsv)


