
def num2mode(para):
    para = para
    if para==0:
        mode = 'Voltage Excitation Mode'
    elif para==1:
        mode = 'Current Excitation Mode'
    return mode

def num2hrng(para):
    para = para
    if para==0:
        htrng = 0
    elif para==1:
        htrng = 3.16e-6
    elif para==2:
        htrng = 100.e-6
    elif para==3:
        htrng = 316.e-6
    elif para==4:
        htrng = 1.00e-3
    elif para==5:
        htrng = 3.16e-3
    elif para==6:
        htrng = 10.0e-3
    elif para==7:
        htrng = 31.6e-3
    elif para==8:
        htrng = 100.e-3
        
    return htrng

def num2exv(para):
    para = para
    if para==1:
        exv = '2.00uV'
    elif para==2:
        exv = '6.32uV'
    elif para==3:
        exv = '20.0uV'
    elif para==4:
        exv = '63.2uV'
    elif para==5:
        exv = '200uV'
    elif para==6:
        exv = '632uV'
    elif para==7:
        exv = '2.00mV'
    elif para==8:
        exv = '6.32mV'
    elif para==9:
        exv = '20.0mV'
    elif para==10:
        exv = '63.2mV'
    elif para==11:
        exv = '200mV'
    elif para==12:
        exv = '632mV'
            
    return exv

def exv2num(para):
    para = para
    if para=='2.00uV':
        num = 1
    elif para=='2uV':
        num = 1
    elif para=='6.32uV':
        num = 2
    elif para=='20.0uV':
        num = 3
    elif para=='20uV':
        num = 3
    elif para=='63.2uV':
        num = 4
    elif para=='200uV':
        num = 5
    elif para=='632uV':
        num = 6
    elif para=='2.00mV':
        num = 7
    elif para=='2mV':
        num = 7
    elif para=='6.32mV':
        num = 8
    elif para=='20.0mV':
        num = 9
    elif para=='20mV':
        num = 9
    elif para=='63.2mV':
        num = 10
    elif para=='200mV':
        num = 11
    elif para=='632mV':
        num = 12
            
    return num

def num2rng(para):
    para = para
    if para==1:
        rng = '2.00mOhm'
    elif para==2:
        rng = '6.32mOhm'
    elif para==3:
        rng = '20.0mOhm'
    elif para==4:
        rng = '63.2mOhm'
    elif para==5:
        rng = '200mOhm'
    elif para==6:
        rng = '632mOhm'
    elif para==7:
        rng = '2.00 Ohm'
    elif para==8:
        rng = '6.32 Ohm'
    elif para==9:
        rng = '20.0 Ohm'
    elif para==10:
        rng = '63.2 Ohm'
    elif para==11:
        rng = '200 Ohm'
    elif para==12:
        rng = '632 Ohm'
    elif para==13:
        rng = '2.00kOhm'
    elif para==14:
        rng = '6.32kOhm'
    elif para==15:
        rng = '20.0kOhm'
    elif para==16:
        rng = '63.2kOhm'
    elif para==17:
        rng = '200kOhm'
    elif para==18:
        rng = '632kOhm'        
    elif para==19:
        rng = '2.00MOhm'
    elif para==20:
        rng = '6.32MOhm'
    elif para==21:
        rng = '20.0MOhm'
    elif para==22:
        rng = '63.2MOhm'

    return rng

def rng2num(para):
    para = para
    if para=='2.00mOhm':
        num = 1
    elif para=='2mOhm':
        num = 1
    elif para=='6.32mOhm':
        num = 2
    elif para=='20.0mOhm':
        num = 3
    elif para=='20mOhm':
        num = 3
    elif para=='63.2mOhm':
        num = 4
    elif para=='200mOhm':
        num = 5
    elif para=='632mOhm':
        num = 6
    elif para=='2.00 Ohm':
        num = 7
    elif para=='2.00Ohm':
        num = 7
    elif para=='2Ohm':
        num = 7
    elif para=='2.0Ohm':
        num = 7
    elif para=='2.0 Ohm':
        num = 7
    elif para=='6.32 Ohm':
        num = 8
    elif para=='6.32Ohm':
        num = 8
    elif para=='20.0 Ohm':
        num = 9
    elif para=='20.0Ohm':
        num = 9
    elif para=='20Ohm':
        num = 9
    elif para=='63.2 Ohm':
        num = 10
    elif para=='63.2Ohm':
        num = 10
    elif para=='200 Ohm':
        num = 11
    elif para=='200Ohm':
        num = 11
    elif para=='632 Ohm':
        num = 12
    elif para=='632Ohm':
        num = 12
    elif para=='2.00kOhm':
        num = 13
    elif para=='2kOhm':
        num = 13
    elif para=='6.32kOhm':
        num = 14
    elif para=='20.0kOhm':
        num = 15
    elif para=='20kOhm':
        num = 15
    elif para=='63.2kOhm':
        num = 16
    elif para=='200kOhm':
        num = 17
    elif para=='632kOhm':
        num = 18
    elif para=='2.00MOhm':
        num = 19
    elif para=='6.32MOhm':
        num = 20
    elif para=='20.0MOhm':
        num = 21
    elif para=='63.2MOhm':
        num = 22
            
    return num






