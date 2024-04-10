''' Juntar las tablas Base y Oficina, guardar la nueva tabla en la carpeta raw data en formato csv'''

import pandas as pd 

excel_path = 'Prueba Técnica. Científico(a) de Datos.xlsx'

base = pd.read_excel(excel_path, sheet_name = 'Base')
oficinas = pd.read_excel(excel_path, sheet_name = 'Oficina')

# Reemplazar el código de la oficina por su nombre en la tabla Base
oficinas = oficinas.set_index('Oficina')['nombreOficina'].to_dict()
base.oficina = base['oficina'].map(oficinas)

# Cambiar el nombre de algunas columnas
col_map = {
    'ocupació' : 'ocupacion',
    'categori' : 'categoria',
    'tiempode' : 'tiempo_desembolso',
    'formapag' : 'forma_de_pago',
    'reestruc' : 'reestructurado',
    'niveledu' : 'nivel_educativo',
    'ingtot' : 'ingreso_total',
    'egrtot' : 'egreso_total',
    'antigemp' : 'antiguedad_empresa',
    'estadoci' : 'estado_civil',
    'tipovivi' : 'tipo_vivienda',
    'tipocont' : 'tipo_contrato', 
    'numerocr' : 'numero_creditos',
    'antigcoo' : 'antiguedad_entidad',
    'default' : 'clase'
}
base = base.rename(columns = col_map)

# Las columnas categoria y Cat son iguales
base = base.drop(columns = 'Cat')

# Guardar como csv
base.to_csv('data/raw/all_data.csv', index = False)
print('Base inicial guardada en la carpeta data exitosamente')