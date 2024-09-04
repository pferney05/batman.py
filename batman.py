#!/Users/fernpa/anaconda3/bin/python

import lxml.etree
import numpy as np
import numpy.linalg
import scipy
import scipy.linalg
import pandas as pd
import warnings
import matplotlib.pyplot as plt


BRANCHING_RATIO_TOLERANCE = 1e-16

class Flux: # n.cm-2.s-1

    def __init__(self, fluxes: np.ndarray, energies: np.ndarray = np.array([1.e-5,2.e7])):
        self.energies = np.array(energies, dtype=float, ndmin=1)
        self.fluxes = np.array(fluxes, dtype=float, ndmin=1) 
        assert np.shape(self.energies)[0] == np.shape(self.fluxes)[0] + 1

    def __str__(self):
        stro = '+-Flux-mesh--+%.4e eV\n'%(self.energies[0]) 
        for i in range(0,np.shape(self.fluxes)[0]):
            stro += '| %.4e |\n'%(self.fluxes[i])
            stro += '+------------+%.4e eV\n'%(self.energies[i+1])
        return stro

class CrossSection: # barn

    def __init__(self,  xsections: np.ndarray, energies: np.ndarray = np.array([1.e-5,2.e7])):
        self.energies = np.array(energies, dtype=float, ndmin=1)
        self.xsections = np.array(xsections, dtype=float, ndmin=1) 
        assert np.shape(self.energies)[0] == np.shape(self.xsections)[0] + 1

    def __str__(self):
        stro = '+-XSection---+%.4e eV\n'%(self.energies[0]) 
        for i in range(0,np.shape(self.xsections)[0]):
            stro += '| %.4e |\n'%(self.xsections[i])
            stro += '+------------+%.4e eV\n'%(self.energies[i+1])
        return stro

class Decay:

    selfTarget = ['sf']

    def __init__(self, type: str = None, target: str = None, branching_ratio: float = None):
        self.type = type
        self.target = target
        self.branching_ratio = branching_ratio
        if not self.type is None:
            self.type = str(self.type)
        if not self.target is None:
            self.target = str(self.target)
        if not self.branching_ratio is None:
            self.branching_ratio = float(self.branching_ratio)

    def get_branching_ratio(self):
        if not self.branching_ratio is None:
            return self.branching_ratio
        else:
            return 1.0

    def __str__(self):
        return f'Decay type: {self.type}, target: {self.target}, branching_ratio: {self.branching_ratio}'

class Reaction:

    def __init__(self, type: str = None, Q: float = None, target: str = None, branching_ratio: float = None, xs : CrossSection = None):
        self.type = type
        self.Q = Q
        self.target = target
        self.branching_ratio = branching_ratio
        if not self.type is None:
            self.type = str(self.type)
        if not self.Q is None:
            self.Q = float(self.Q)
        if not self.target is None:
            self.target = str(self.target)
        if not self.branching_ratio is None:
            self.branching_ratio = float(self.branching_ratio)
        self.xs = xs

    def __str__(self):
        if self.xs is None:
            return f'Reaction type: {self.type}, target: {self.target}, branching_ratio: {self.branching_ratio}, Q: {self.Q}, xs:{self.xs}'
        else:
            return f'Reaction type: {self.type}, target: {self.target}, branching_ratio: {self.branching_ratio}, Q: {self.Q}, xs:{self.xs.xsections}'

    def set_xsection(self, xs: CrossSection):
        assert type(xs) is CrossSection
        self.xs = xs

    def get_branching_ratio(self):
        if not self.branching_ratio is None:
            return self.branching_ratio
        else:
            return 1.0

class FissionYield(dict):

    def __init__(self, energy: float = None, *args, **kwargs):
        self.energy = energy
        if type(energy) is None:
            self.energy = float(self.energy)
        super().__init__(*args, **kwargs)

    def set_from_list(self, nuclides: list, yields: list):
        assert len(nuclides) == len(yields)
        for i in range(0,len(yields)):
            nuc = nuclides[i]
            fyield = yields[i]
            if fyield != 0.:
                self[nuc] = fyield

class Nuclide:

    def __init__(self, name: str):
        self.name = name
        self.l = None
        self.reactions = []
        self.decays = []
        self.yields = []

    def __getitem__(self, item):
        itemList = []
        for reac in self.reactions:
            if item == reac.type:
                itemList.append(reac)
        return itemList

    def add_decay(self, *args, **kwargs):
        decay = Decay(*args,**kwargs)
        self.decays.append(decay)

    def add_reaction(self, *args, **kwargs):
        reaction = Reaction(*args, **kwargs)
        self.reactions.append(reaction)
    
    def add_fyields(self, xmldata: lxml.etree._Element):
        if 'parent' in xmldata.attrib.keys():
            self.yields = str(xmldata.attrib['parent'])
        else:
            for i in range(1,len(xmldata)):
                fissionyield = FissionYield()
                fissionyield.energy = float(xmldata[i].attrib['energy'])
                nuclides = [str(x) for x in xmldata[i][0].text.split(' ')]
                yields = [float(x) for x in xmldata[i][1].text.split(' ')]
                fissionyield.set_from_list(nuclides, yields)
                self.yields.append(fissionyield)

    def set_xsections(self, reac: str, xs: CrossSection):
        assert type(xs) is CrossSection
        if np.any(xs.xsections > 0.):
            for reaction in self[reac]:
                reaction.set_xsection(xs)

    def set_decay_const(self, xmldict: dict):
        if 'half_life' in xmldict.keys():
            self.l = np.log(2) / float(xmldict['half_life'])
        else:
            self.l = 0.

    def set_from_xml(self, xmlnuc: lxml.etree._Element):
        self.set_decay_const(xmlnuc.attrib)
        for xmlinter in xmlnuc:
            if xmlinter.tag == 'decay':
                self.add_decay(**xmlinter.attrib)
            elif xmlinter.tag == 'reaction':
                self.add_reaction(**xmlinter.attrib)
            elif xmlinter.tag == 'neutron_fission_yields':
                self.add_fyields(xmlinter)
            elif xmlinter.tag == 'source':
                pass
            else:
                warnings.warn('This element tag is unknown and will not be loaded: %s'%(xmlinter.tag))

    def get_connected_nuclides(self, fissionProducts = True):
        nuclideList = []
        isfission = False
        # Decay products
        for decay in self.decays:
            if (not decay.target in nuclideList) and (not decay.type in Decay.selfTarget):
                nuclideList.append(decay.target)
            elif decay.type == 'sf' and fissionProducts:
                isfission = True
        # Reaction products
        for reaction in self.reactions:
            if (not reaction.target in nuclideList) and (reaction.type != 'fission') and (not reaction.xs is None):
                nuclideList.append(reaction.target)
            elif reaction.type == 'fission' and fissionProducts and (not reaction.xs is None):
                isfission = True
        # Fission products
        if isfission:
            if type(self.yields) is str:
                nuclideList.append(self.yields)
            else:
                for fyield in self.yields:
                    for nucname in fyield.keys():
                        if not nucname in nuclideList:
                            nuclideList.append(nucname)
        return nuclideList
    
    def get_total_branching(self, type: str):
        total_branching = 0.
        for reaction in self.reactions:
            if reaction.type == type and reaction.branching_ratio is None:
                total_branching = 1.0
            elif reaction.type == type:
                total_branching += reaction.branching_ratio
        return total_branching
    


class Chain(dict):

    def __init__(self, chainpath: str = './ENDF_default_chain.xml', energies: np.ndarray = np.array([1.e-5,2.e7]), *args, **kwargs):
        self.energies = np.array(energies, dtype=float, ndmin=1)
        super().__init__(*args, **kwargs)
        self.load_chain(chainpath)
          
    def fetch_nuclide(self, name: str):
        if not name in self.keys():
            self[name] = Nuclide(name)
        return self[name]

    def load_chain(self, xmlname: str):
        tree = lxml.etree.parse(xmlname)
        root = tree.getroot()
        for xmlnuc in root:
            nuclide = self.fetch_nuclide(xmlnuc.attrib['name'])
            nuclide.set_from_xml(xmlnuc)
        
    def load_cross_sections(self, csvname: str):
        data = pd.read_csv(csvname)
        for i in range(0,data.shape[0]):
            args = np.array(data.loc[i])
            nucname, reac = args[:2]
            xsections = np.array([float(x) for x in args[3:]])
            xs = CrossSection(xsections, self.energies)
            nuclide = self.fetch_nuclide(nucname)
            nuclide.set_xsections(reac, xs)

    def get_connected_nuclides(self, nuclideName: str, fissionProducts: bool = True, nuclideList = []):
        if nuclideName in self.keys():
            nuclideList.append(nuclideName)
            nuclide = self[nuclideName]
            productsList = nuclide.get_connected_nuclides(fissionProducts=fissionProducts)
            for product in productsList:
                if not product in nuclideList:
                    connectedNuclides = self.get_connected_nuclides(product, fissionProducts, nuclideList)
                    for cNuclide in connectedNuclides:
                        if cNuclide not in nuclideList:
                            nuclideList.append(cNuclide)
        return nuclideList

    def get_yields(self, nuclideName: str):
        nuclide = self[nuclideName]
        nfyields = nuclide.yields
        if type(nuclide.yields) == str:
            nfyields = self[nuclide.yields].yields
        return nfyields
    
    def interp_yields(self, nuclideName: str, energy: float = 0.0253):
        yields = self.get_yields(nuclideName)
        rdist = +np.inf
        for i, fyield in enumerate(yields):
            # This is the nearest logarithmic energy. Indeed, if a neutron has an energy of 0.5 MeV and you have data for 0.0253 eV and 2 Mev, 2 MeV makes more physical sense.
            dist = abs(np.log(fyield.energy) - np.log(energy)) 
            if i==0 or dist < rdist:
                myYield = fyield
                rdist = dist
        return myYield


class Step:

    def __init__(self, chain: Chain, t0: float = 0., endTime: float =+np.inf, fissionProducts: bool = True):
        self.nuclideIndex = {}
        self.chain = chain
        self.t0 = t0
        self.X0 = None
        self.A = None
        self.P = None
        self.iP = None
        self.D = None
        self.tEnd = endTime
        self.fissionProducts = fissionProducts

    def init_index(self, nuclideList: list, chain: Chain):
        nuclideIndex = {}
        i=0
        for nuclide in nuclideList:
            connectedNuclideList = chain.get_connected_nuclides(nuclide, self.fissionProducts)
            for cNuclide in connectedNuclideList:
                if not cNuclide in nuclideIndex.keys():
                    nuclideIndex[cNuclide] = i
                    i += 1
        return nuclideIndex

    def init_quantity(self, iniQty: dict):
        X0 = np.zeros(len(self.nuclideIndex))
        for nuc, quantity in iniQty.items():
            i = self.nuclideIndex[nuc]
            X0[i] = quantity
        return X0

    def init_matrix(self, flux: Flux):
        nNuc = len(self.nuclideIndex)
        operator = np.zeros(shape=(nNuc,nNuc))
        # Decay interactions
        for nuc, i in self.nuclideIndex.items():
            nuclide = self.chain[nuc]
            lmda = self.chain[nuc].l
            operator[i,i] += -lmda
            total_branching = 0.
            for decay in nuclide.decays:
                bratio = decay.get_branching_ratio()
                total_branching += bratio
                if decay.type=='sf' and self.fissionProducts:
                    nfyields = self.chain.interp_yields(nuc) # Spontaneous fissions are considered with a yield equivalent to a 0.253 eV neutron induced fission.
                    for newnuc in nfyields.keys():
                        thisYield = float(nfyields[newnuc])
                        it = self.nuclideIndex[newnuc]
                        operator[it,i] += +lmda * thisYield * bratio
                elif decay.type!='sf':
                    target = decay.target
                    it = self.nuclideIndex[target]
                    operator[it,i] += +lmda * bratio
            if abs(total_branching - 1.) > BRANCHING_RATIO_TOLERANCE and len(nuclide.decays)!=0:
                warnings.warn('Decay branching ratio for %s, %s should add up to one but equal: %s'%(nuc, decay.type, total_branching))
        # Neutron interactions
        for nuc, i in self.nuclideIndex.items():
            nuclide = self.chain[nuc]
            for reaction in nuclide.reactions:
                bratio = reaction.get_branching_ratio()
                if not reaction.xs is None:
                    operator[i,i] += -bratio * np.sum(reaction.xs.xsections * flux.fluxes) / 10.**24
                    if reaction.type =='fission' and self.fissionProducts:
                        for j in range(0,np.shape(flux.fluxes)[0]):
                            energy = (flux.energies[j]*flux.energies[j+1])**0.5
                            nfyields = self.chain.interp_yields(nuc, energy)
                            for newnuc in nfyields.keys():
                                thisYield = float(nfyields[newnuc])
                                it = self.nuclideIndex[newnuc]
                                operator[it,i] += +bratio * thisYield * (reaction.xs.xsections[j] * flux.fluxes[j]) / 10.**24
                    elif reaction.type !='fission':
                        target = reaction.target
                        it = self.nuclideIndex[target]
                        operator[it,i] += +bratio * np.sum(reaction.xs.xsections * flux.fluxes) / 10.**24
                total_branching = nuclide.get_total_branching(reaction.type)
                if abs(total_branching - 1.) > BRANCHING_RATIO_TOLERANCE and len(nuclide.reactions)!=0:
                    warnings.warn('Reaction Branching ratio for %s, %s should add up to one but equal: %s'%(nuc, reaction.type, total_branching))
        return operator

    def set_matrix(self, flux: Flux):
        self.A = self.init_matrix(flux)
        eigvalues, self.P = numpy.linalg.eig(self.A)
        self.iP = numpy.linalg.inv(self.P)
        self.D = np.diag(eigvalues)

    def solve(self,t):
        assert self.t0 <= t and t <= self.tEnd 
        eD = scipy.linalg.expm((t-self.t0)*self.D)
        # PeD = np.matmul(self.P,eD)
        # iPX0 = np.matmul(self.iP,self.X0)
        # PeDiPX0 = np.matmul(PeD,iPX0)
        # PeD = np.matmul(self.P,eD)
        # PeDiP = np.matmul(PeD,self.iP)
        # PeDiPX0 = np.matmul(PeDiP,self.X0)
        iPX0 = np.matmul(self.iP,self.X0)
        eDiPX0 = np.matmul(eD,iPX0)
        PeDiPX0 = np.matmul(self.P,eDiPX0)
        return np.abs(PeDiPX0)

    def solve_step(self, tpoints, method = 'RK45', atol = 1.e-3, rtol=1.e-6):
        def func(t, X):
            return np.matmul(self.A, X)
        sol =  scipy.integrate.solve_ivp(func, [self.t0 , self.tEnd], self.X0, method=method, t_eval = tpoints, atol=atol, rtol=rtol)
        return sol.y

    @classmethod
    def initial(cls, iniQty: dict, flux: Flux, chain: Chain, t0: float = 0., endTime: float =+np.inf, fissionProducts: bool = True):
        thisStep = cls(chain, t0, endTime, fissionProducts)
        thisStep.nuclideIndex = thisStep.init_index(iniQty.keys(), chain)
        thisStep.X0 = thisStep.init_quantity(iniQty)
        thisStep.set_matrix(flux)
        return thisStep

    @classmethod
    def after(cls, previousStep: 'Step', flux: Flux, endTime: float =+np.inf):
        t0 = previousStep.tEnd
        thisStep = cls(previousStep.chain, t0, endTime, previousStep.fissionProducts)
        thisStep.nuclideIndex = previousStep.nuclideIndex
        thisStep.X0 = previousStep.solve(t0)
        thisStep.set_matrix(flux)
        return thisStep

    

class Sequence(list):
    
    def __init__(self, chain: Chain, fissionProducts = True, *args, **kwargs):
        self.chain = chain
        self.nuclideIndex = {}
        self.iniQty = {}
        self.fissionProducts = fissionProducts
        super().__init__(*args, **kwargs)

    def set_init_quantity(self, iniQty: dict):
        if 'C0' in iniQty.keys():
            C12 = 0.9894 * iniQty['C0']
            C13 = 0.0106 * iniQty['C0']
            del iniQty['C0']
            iniQty['C12'] = C12
            iniQty['C13'] = C13
        self.iniQty = iniQty

    def add_initial(self, tStart: float = 0., tEnd: float = +np.inf, flux: Flux = Flux([0.])):
        step = Step.initial(self.iniQty, flux, self.chain, tStart, tEnd, self.fissionProducts)
        self.nuclideIndex = step.nuclideIndex
        self.append(step)

    def add_after(self, step: Step, tEnd: float = +np.inf, flux: Flux = Flux([0.])):
        self.append(Step.after(step, flux, endTime=tEnd))

    def add_after_last(self, tEnd: float = +np.inf, flux: Flux = Flux([0.])):
        step = self[-1]
        self.append(Step.after(step, flux, endTime=tEnd))

    def check_steps(self):
        allStart = None
        allEnd = None
        for step in self:
            if allStart is None:
                allStart = step.t0
            elif allStart > step.t0:
                allStart = step.t0
            if allEnd is None:
                allEnd = step.tEnd
            elif allEnd < step.tEnd:
                allEnd = step.tEnd
        for step in self:
            if step.t0!=allStart:
                isJoint = False
                for otherstep in self:
                    if otherstep.tEnd == step.t0:
                        isJoint = True
                        break
                if not isJoint:
                    raise ValueError("A step starts at %s, but does not follow any other steps"%(step.t0))

    def get_step(self, t: float):
        myStep = None
        for step in self:
            if step.t0 < t and t <= step.tEnd:
                myStep = step
                break
        if myStep is None:
            for step in self:
                if step.t0 == t:
                    myStep = step
                    break
        return myStep

    def solve_all(self, tpoints: list):
        nNuc = len(self[0].nuclideIndex)
        nDot = len(tpoints)
        output = np.zeros((nNuc, nDot))
        for i, t in enumerate(tpoints):
            print(i)
            step = self.get_step(t)
            output[:,i] = step.solve(t)
        return output
    
    def solve_activity(self, tpoints: list):
        quantities = self.solve_all(tpoints)
        totalActivity = np.zeros(np.shape(quantities)[1])
        activity = np.zeros_like(quantities)
        for nuc in self.nuclideIndex.keys():
            index = self.nuclideIndex[nuc]
            activity[index,:] = self.chain[nuc].l * quantities[index,:]
            totalActivity += activity[index,:]
        return activity, totalActivity
    
    def _plot(self, title='', xLabel='', yLabel='', xmin = None, xmax=None, ymin = None, ymax=None, ppp=150, xlogScale = False, ylogScale = True):
        num = len(plt.get_fignums())
        fig=plt.figure(num=num, figsize=[6.4,4.8],dpi=ppp, frameon=True)
        ax=fig.add_axes([0.12,0.12,0.75,0.75])
        ax.set_title(title, fontsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if xlogScale:
            ax.set_xscale('log')
        if ylogScale:
            ax.set_yscale('log')
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.tick_params(which='major',direction='out', length=8, width=1)
        ax.tick_params(which='minor',direction='out', length=4, width=0.8)
        ax.minorticks_on()
        ax.grid(which="major",zorder=0,color='#C0C0C0')
        ax.grid(which="minor",zorder=-1,color='#E0E0E0')
        return fig, ax

    def plot_quantity(self, tpoints: list, nuclideList: list = 'all', title='', xLabel='', yLabel='', xmin = None, xmax=None, ymin = None, ymax=None, ls='-', marker='', lw = 0.75, cmap='viridis', ppp=150, xlogScale = False, ylogScale = True, xscale=1.):
        nuclideIndex = self.nuclideIndex
        output = self.solve_all(tpoints)
        if nuclideList=='all':
            nuclideList=list(nuclideIndex.keys())
        iList = []
        for nuc in nuclideList:
            iList.append(nuclideIndex[nuc])
        if xmin is None:
            xmin = np.min(tpoints)/xscale
            if xmin == 0. and xlogScale:
                xmin=1./xscale
        if xmax is None:
            xmax = np.max(tpoints)/xscale
        if ymin is None:
            if ylogScale:
                ymin = 1.
            else:
                ymin=0.
        if ymax is None:
            ymax = np.max(output[iList,:])
        fig, ax = self._plot(title, xLabel, yLabel, float(xmin), float(xmax), float(ymin), float(ymax), ppp, xlogScale, ylogScale)
        colormap = plt.get_cmap(cmap)
        colors = colormap(np.linspace(0,1,len(nuclideList)))
        for i, nuc in enumerate(nuclideList):
            col = colors[i]
            index = nuclideIndex[nuc]
            yVector = output[index,:]
            line,=ax.plot(tpoints/xscale,yVector,ls=ls,linewidth=lw,marker=marker,markersize=4,zorder=3,color=col)
            line.set_label(nuc)
        ax.legend()
        return fig, ax
    
    def plot_activity(self, tpoints: list, nuclideList: list = 'all', title='', xLabel='', yLabel='', xmin = None, xmax=None, ymin = None, ymax=None, ls='-', marker='', lw = 0.75, cmap='viridis', ppp=150, xlogScale = False, ylogScale = True, xscale=1.):
        nuclideIndex = self.nuclideIndex
        output = self.solve_all(tpoints)
        if nuclideList=='all':
            nuclideList=list(nuclideIndex.keys())
        iList = []
        for nuc in nuclideList:
            iList.append(nuclideIndex[nuc])
        if xmin is None:
            xmin = np.min(tpoints)/xscale
            if xmin == 0. and xlogScale:
                xmin=1./xscale
        if xmax is None:
            xmax = np.max(tpoints)/xscale
        if ymin is None:
            if ylogScale:
                ymin = 1.
            else:
                ymin=0.
        if ymax is None:
            ymax = np.max(output[iList,:])
        fig, ax = self._plot(title, xLabel, yLabel, float(xmin), float(xmax), float(ymin), float(ymax), ppp, xlogScale, ylogScale)
        colormap = plt.get_cmap(cmap)
        colors = colormap(np.linspace(0,1,len(nuclideList)))
        activity, totalActivity = self.solve_activity(tpoints)
        line,=ax.plot(tpoints/xscale,totalActivity,ls=ls,linewidth=2*lw,marker=marker,markersize=4,zorder=3,color='#000000')
        line.set_label('total')
        for i, nuc in enumerate(nuclideList):
            col = colors[i]
            index = nuclideIndex[nuc]
            yVector = activity[index,:]
            line,=ax.plot(tpoints/xscale,yVector,ls=ls,linewidth=lw,marker=marker,markersize=4,zorder=3,color=col)
            line.set_label(nuc)
        ax.legend()
        return fig, ax

    @staticmethod
    def show():
        plt.show()

class Continuous:

    def __init__(self, chain, time, flux, iniQty, fission = True):
        self.chain = chain
        self.time = time 
        self.flux = flux
        step0= Step(chain,time[0],time[-1], fissionProducts=fission)
        step1 = Step(chain,time[0],time[-1], fissionProducts=fission)
        step0.nuclideIndex = step0.init_index(iniQty.keys(), chain)
        self.nuclideIndex = step0.nuclideIndex
        step0.X0=step0.init_quantity(iniQty)
        self.X0 = step0.X0
        step1.init_matrix(1.)
        mat0 = step0.init_matrix(Flux(0.))
        mat1 = step0.init_matrix(Flux(1.))
        self.A = mat1 - mat0
        self.B = mat0
        

    def get_power(self, t):
        return np.interp(t, self.time, self.flux)
    
    def solve(self, tpoints, method = 'Radau', atol = 1.e-3, rtol = 1.e-6):
        def _func(t, X):
            f = self.get_power(t)
            M = self.B + f * self.A
            return np.matmul(M, X)
        def _jac(t, X):
            f = self.get_power(t)
            M = self.B + f * self.A
            return M

        tmin = self.time[0]
        tmax = self.time[-1]

        if method in ['RK45','RK23','DOP853']:
            sol = scipy.integrate.solve_ivp(_func, [tmin, tmax], self.X0, method=method, t_eval=tpoints, atol = atol, rtol = rtol)
        else:
            sol = scipy.integrate.solve_ivp(_func, [tmin, tmax], self.X0, method=method, t_eval=tpoints, atol = atol, rtol = rtol, jac = _jac)

        return sol.y
    
    def solve_activity(self, tpoints, method = 'Radau', atol = 1.e-3, rtol = 1.e-6):
        quantities = self.solve(tpoints, method, atol, rtol)
        totalActivity = np.zeros(np.shape(quantities)[1])
        activity = np.zeros_like(quantities)
        for nuc in self.nuclideIndex.keys():
            index = self.nuclideIndex[nuc]
            activity[index,:] = self.chain[nuc].l * quantities[index,:]
            totalActivity += activity[index,:]
        return activity, totalActivity

    def _plot(self, title='', xLabel='', yLabel='', xmin = None, xmax=None, ymin = None, ymax=None, ppp=150, xlogScale = False, ylogScale = True):
        num = len(plt.get_fignums())
        fig=plt.figure(num=num, figsize=[6.4,4.8],dpi=ppp, frameon=True)
        ax=fig.add_axes([0.12,0.12,0.75,0.75])
        ax.set_title(title, fontsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if xlogScale:
            ax.set_xscale('log')
        if ylogScale:
            ax.set_yscale('log')
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.tick_params(which='major',direction='out', length=8, width=1)
        ax.tick_params(which='minor',direction='out', length=4, width=0.8)
        ax.minorticks_on()
        ax.grid(which="major",zorder=0,color='#C0C0C0')
        ax.grid(which="minor",zorder=-1,color='#E0E0E0')
        return fig, ax

    def plot_quantity(self, tpoints: list, nuclideList: list = 'all', title='', xLabel='', yLabel='', xmin = None, xmax=None, ymin = None, ymax=None, ls='-', marker='', lw = 0.75, cmap='viridis', ppp=150, xlogScale = False, ylogScale = True, xscale=1.):
        nuclideIndex = self.nuclideIndex
        output = self.solve(tpoints)
        if nuclideList=='all':
            nuclideList=list(nuclideIndex.keys())
        iList = []
        for nuc in nuclideList:
            iList.append(nuclideIndex[nuc])
        if xmin is None:
            xmin = np.min(tpoints)/xscale
            if xmin == 0. and xlogScale:
                xmin=1./xscale
        if xmax is None:
            xmax = np.max(tpoints)/xscale
        if ymin is None:
            if ylogScale:
                ymin = 1.
            else:
                ymin=0.
        if ymax is None:
            ymax = np.max(output[iList,:])
        fig, ax = self._plot(title, xLabel, yLabel, float(xmin), float(xmax), float(ymin), float(ymax), ppp, xlogScale, ylogScale)
        colormap = plt.get_cmap(cmap)
        colors = colormap(np.linspace(0,1,len(nuclideList)))
        for i, nuc in enumerate(nuclideList):
            col = colors[i]
            index = nuclideIndex[nuc]
            yVector = output[index,:]
            line,=ax.plot(tpoints/xscale,yVector,ls=ls,linewidth=lw,marker=marker,markersize=4,zorder=3,color=col)
            line.set_label(nuc)
        ax.legend()
        return fig, ax
    
    def plot_activity(self, tpoints: list, nuclideList: list = 'all', title='', xLabel='', yLabel='', xmin = None, xmax=None, ymin = None, ymax=None, ls='-', marker='', lw = 0.75, cmap='viridis', ppp=150, xlogScale = False, ylogScale = True, xscale=1.):
        nuclideIndex = self.nuclideIndex
        activity, totalActivity = self.solve_activity(tpoints)
        if nuclideList=='all':
            nuclideList=list(nuclideIndex.keys())
        iList = []
        for nuc in nuclideList:
            iList.append(nuclideIndex[nuc])
        if xmin is None:
            xmin = np.min(tpoints)/xscale
            if xmin == 0. and xlogScale:
                xmin=1./xscale
        if xmax is None:
            xmax = np.max(tpoints)/xscale
        if ymin is None:
            if ylogScale:
                ymin = 1.
            else:
                ymin=0.
        if ymax is None:
            ymax = np.max(totalActivity[iList,:])
        fig, ax = self._plot(title, xLabel, yLabel, float(xmin), float(xmax), float(ymin), float(ymax), ppp, xlogScale, ylogScale)
        colormap = plt.get_cmap(cmap)
        colors = colormap(np.linspace(0,1,len(nuclideList)))
        activity, totalActivity = self.solve_activity(tpoints)
        line,=ax.plot(tpoints/xscale,totalActivity,ls=ls,linewidth=2*lw,marker=marker,markersize=4,zorder=3,color='#000000')
        line.set_label('total')
        for i, nuc in enumerate(nuclideList):
            col = colors[i]
            index = nuclideIndex[nuc]
            yVector = activity[index,:]
            line,=ax.plot(tpoints/xscale,yVector,ls=ls,linewidth=lw,marker=marker,markersize=4,zorder=3,color=col)
            line.set_label(nuc)
        ax.legend()
        return fig, ax
