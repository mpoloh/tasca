import  pandas  as  pd  
datos  =  { 'isla':  ['Mallorca',  'Menorca',  'Ibiza',  'Formentera'],  'superficie':  [3620,  692,  
577,  83],  'poblacio':  [ 923608,  94885,  147914,  
11708]
 }  
df  =  pd.DataFrame(datos)  
print("Superficie  total:",  df['superficie'].sum(),  "km2")