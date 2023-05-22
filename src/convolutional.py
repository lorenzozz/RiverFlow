from PIL import Image
import Config


gen_path = Config.RIVERDATAROOT+'/SatelliteData/SAT-DATA-2022-01-01.png'
b = Image.open(gen_path).convert('RGBA')
print(b.show())
