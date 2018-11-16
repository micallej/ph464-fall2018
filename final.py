#lighting the hearth
import numpy
import matplotlib.pyplot
import pandas
import csv


#preperations
SuperConductorsCSV = numpy.loadtxt('predict_tc-master/train.csv',delimiter=',',skiprows=1)
SuperConductorsFeatures = numpy.array(['number_of_elements','mean_atomic_mass','weighted_mean_atomic_mass','gmean_atomic_mass','weighted_gmean_atomic_mass','entropy_atomic_mass','weighted_entropy_atomic_mass','range_atomic_mass','weighted_range_atomic_mass','standard_atomic_mass','weighted_standard_atomic_mass',
'mean_fie','weighted_mean_fie','gmean_fie','weighted_gmean_fie','entropy_fie','weighted_entropy_fie','range_fie','weighted_range_fie','standard_fie','weighted_standard_fie',
'mean_atomic_radius','weighted_mean_atomic_radius','gmean_atomic_radius','weighted_gmean_atomic_radius','entropy_atomic_radius','weighted_entropy_atomic_radius','range_atomic_radius','weighted_range_atomic_radius','standard_atomic_radius','weighted_standard_atomic_radius',
'mean_density','weighted_mean_density','gmean_density','weighted_gmean_density','entropy_density','weighted_entropy_density','range_density','weighted_range_density','standard_density','weighted_standard_density',
'mean_electron_affinity','weighted_mean_electron_affinity','gmean_electron_affinity','weighted_gmean_electron_affinity','entropy_electron_affinity','weighted_entropy_electron_affinity','range_electron_affinity','weighted_range_electron_affinity','standard_electron_affinity','weighted_standard_electron_affinity',
'mean_fusion_heat','weighted_mean_fusion_heat','gmean_fusion_heat','weighted_gmean_fusion_heat','entropy_fusion_heat','weighted_entropy_fusion_heat','range_fusion_heat','weighted_range_fusion_heat','standard_fusion_heat','weighted_standard_fusion_heat',
'mean_thermal_conductivity','weighted_mean_thermal_conductivity','gmean_thermal_conductivity','weighted_gmean_thermal_conductivity','entropy_thermal_conductivity','weighted_entropy_thermal_conductivity','range_thermal_conductivity','weighted_range_thermal_conductivity','standard_thermal_conductivity','weighted_standard_thermal_conductivity',
'mean_valence','weighted_mean_valence','gmean_valence','weighted_gmean_valence','entropy_valence','weighted_entropy_valence','range_valence','weighted_range_valence','standard_valence','weighted_standard_valence',
'critical_temperature'])
SuperConductorsDataFrame = pandas.DataFrame(data=SuperConductorsCSV,columns=SuperConductorsFeatures)
SuperConductorsData = SuperConductorsDataFrame.loc[:,SuperConductorsDataFrame.columns !='critical_temperature']
SuperConductorsTarget = SuperConductorsDataFrame.loc[:,'critical_temperature']
#visualize

for Name in SuperConductorsData.columns:
   
    
    matplotlib.pyplot.figure()
    matplotlib.pyplot.scatter(SuperConductorsData.loc[:,Name],SuperConductorsTarget[:])
    matplotlib.pyplot.xlabel(Name)
    matplotlib.pyplot.ylabel("critical temperature")    
    

#show plots
matplotlib.pyplot.show()
