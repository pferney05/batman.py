#!/Users/fernpa/anaconda3/bin/python

import batman

chain = batman.Chain()
print(chain['U235'].get_connected_nuclides(False))
#print(chain['U235'].get_yield(1.))
nucList = chain.get_connected_nuclides('U238',True)
print(len(nucList))
nucDict = batman.Step({'U235':1.,'U238':1.},batman.Flux([1.]),chain).nuclideIndex

for key, value in nucDict.items():
    stro = "{0: >9s} : {1}".format(key,value)
    print(stro)

# flux = batman.Flux([1e-5,1.,2e7],[0.93 * 5.e13,0.07 * 5.e13])
# xs = batman.CrossSection([1e-5,1.,2e7],[580.,1.])
# print(flux)
# print(xs)
