from DataOrganizer import DataFormatReader

__licence__ = 'Comune di Val Sesia, agenzia federale'
__version__ = '0.1'
__dependencies__ = ['numpy 1.20', 'tensorflow .', 'matplotlib .']

if __name__ == '__main__':

    make_file = "/your/path"

    Make_File = DataFormatReader(make_file)
    Make_File.interpret()
