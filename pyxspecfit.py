from xspec import *
class DataIO:
    def __init__(self, savehdf5="dummy.hdf5"):
        self.savehdf5 = savehdf5
    
    def set_spec(self, spec, rmf, arf, multiresp, bgd=None, datagroup=1, spectrumgroup=1):
        """
        Set spectrum and response file
        """
        AllData(f'{datagroup}:{spectrumgroup} {spec}')
        s                  = AllData(spectrumgroup)
        s.response         = rmf
        s.response.arf     = arf
        s.multiresponse[1] = multiresp
        s.background       = bgd


class SpecModel:
    def __init__(self):
        pass

    def set_model(self, model_str:str):
        self.model = Model(self.model_str)

class SpecFit:
    def __init__(self, abundance="lpgs", atomdb_version="3.0.9"):
        self.abundance = abundance
        self.atomdb_version = atomdb_version
        Xset.parallel.leven = 6
        Xset.parallel.error = 6
        Xset.parallel.steppar = 6

    
class SpecPlot:
    def __init__(self):
        pass


class CLI:
    def __init__(self):
        pass

    def fitting_spectrum(self):
        data = DataIO(savehdf5="dummy.hdf5")
        data.set_spec(spec="spec.pi", rmf="rmf.rmf", arf="arf.arf", multiresp="multiresp.rmf")
        model = SpecModel()
        model.set_model()
        fit = SpecFit()
        fit.fit()
