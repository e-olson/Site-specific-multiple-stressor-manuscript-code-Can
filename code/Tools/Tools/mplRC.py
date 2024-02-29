import matplotlib as mpl

def setRC(rcs):
    mpl.rc('figure', titlesize=rcs.fs) # font size doubled
    mpl.rc('figure', figsize=rcs.fsz) # fig size doubled (half standard page)
    mpl.rc('figure', dpi=100) # dpi halved # not anymore; want lower res notebooks for smaller files
    mpl.rc('savefig', dpi=400) #changed to desired output dpi
    mpl.rc('axes', titlesize=rcs.fs)
    mpl.rc('axes', labelsize=rcs.fs)
    mpl.rc('xtick', labelsize=rcs.fs)
    mpl.rc('ytick', labelsize=rcs.fs)
    mpl.rc('legend', fontsize=rcs.fs)
    mpl.rc('legend', numpoints=1)
    mpl.rc('lines', linewidth=rcs.lw)
    mpl.rc('lines', markersize=rcs.ms)
    #mpl.rc('text', usetex=True)
    #mpl.rc('text.latex', preamble = r'''
    # \usepackage{txfonts}
    # \usepackage{lmodern}
    # ''')
    mpl.rc('font', size=rcs.fs, family='sans-serif', weight='normal', style='normal')
    mpl.rc('figure', max_open_warning =0 )
 
class paperRC:
    fs=16       # doubled
    fsz=(16,10) # doubled
    lw=1
    ms=8
    def __init__(self):
        setRC(self)
        #print(f'mpl default settings: fs={self.fs}, lw={self.lw}, ms={self.ms}')

class paperRC2:
    fs=8
    fsz=(8,5)
    lw=1
    ms=8
    def __init__(self):
        setRC(self)
        #print(f'mpl default settings: fs={self.fs}, lw={self.lw}, ms={self.ms}')


class slidesRC:
    fs=32
    fsz=(16,10) # doubled
    lw=2
    ms=12
    def __init__(self):
        setRC(self)
        #print(f'mpl default settings: fs={self.fs}, lw={self.lw}, ms={self.ms}')

class slidesRC2:
    fs=18
    fsz=(16,10) # doubled
    lw=2
    ms=12
    def __init__(self):
        setRC(self)
        #print(f'mpl default settings: fs={self.fs}, lw={self.lw}, ms={self.ms}')

class posterRC:
    fs=20
    fsz=(16,10) # doubled
    lw=1
    ms=10
    def __init__(self):
        setRC(self)
        #print(f'mpl default settings: fs={self.fs}, lw={self.lw}, ms={self.ms}')

class notebookRC:
    fs=20
    fsz=(16,10) # doubled
    lw=1
    ms=10
    def __init__(self):
        from Tools import notebookDisplay
        setRC(self)
        #print(f'mpl default settings: fs={self.fs}, lw={self.lw}, ms={self.ms}')

class notebookRC2:
    fs=8
    fsz=(7.5,5) # doubled
    lw=.5
    ms=4
    def __init__(self):
        from Tools import notebookDisplay
        setRC(self)
        #print(f'mpl default settings: fs={self.fs}, lw={self.lw}, ms={self.ms}')

class notebookRC3:
    fs=12
    fsz=(7.5,5) # doubled
    lw=1
    ms=6
    def __init__(self):
        from Tools import notebookDisplay
        setRC(self)
        #print(f'mpl default settings: fs={self.fs}, lw={self.lw}, ms={self.ms}')

def showRC(mmm):
    print(f'mpl default settings: fs={mmm.fs}, lw={mmm.lw}, ms={mmm.ms}')
    return

# default style:
note=notebookRC2();
#paper=paperRC()
