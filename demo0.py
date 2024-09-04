#!/Users/fernpa/anaconda3/bin/python

import numpy as np
import batman


startingQuantity = {
    'U238':0.993 * 6.02e23,
    'U235':0.007 * 6.02e23
    }

flux = batman.Flux(fluxes=0.)
chain = batman.Chain()

time = np.array([0.,1e17])
power = np.array([0.,0.])
prob = batman.Continuous(chain, time, power, startingQuantity, fission=False)

tpoints = np.logspace(0,17,num=100)
nuclideList = ['U238','U234','Th230','Pb206','U235','Pa231','Pb207']

activity, totalActivity = prob.solve_activity(tpoints)

prob.plot_quantity(tpoints,nuclideList,ymax=10**24,title='Qty vs. Time',xLabel='Time(years)',yLabel='Quantity (atoms)',xlogScale=True,cmap='jet',xscale=3600.*24*365.25)
prob.plot_activity(tpoints,nuclideList,ymin=10**-12,title='Activity vs. Time',xLabel='Time(years)',yLabel='Activity (Bq)',ymax=10**9,xlogScale=True,cmap='jet',xscale=3600.*24*365.25)

batman.Sequence.show()
