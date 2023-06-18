import struct
import numpy as np
from PIL import Image
from etl8_mappings import HIRAGANA_READING_TO_ORDER

RECORD_SIZE = 8199
NUM_DATASETS = 5
NUM_CATEGORIES = 956

def read_ETL8G_record(f):
    '''
    From the ETL8 spec:

    2. Contents of Logical Record (8199 bytes)

    --------------------------------------------------------------------------------------------
    |             |Number|        |                                                              |
    |     Byte    |  of  |  Type  |            Contents of Logical Record                        |
    |   Position  | Bytes|        |                                                              |
    |============================================================================================|
    |    1 -    2 |    2 | Integer| Serial Sheet Number (greater than or equal to 1)             |
    |    3 -    4 |    2 | Binary | JIS Kanji Code (JIS X 0208)                                  |
    |    5 -   12 |    8 | ASCII  | JIS Typical Reading ( ex. "AI.MEDER" )                       |
    |I  13 -   16 |    4 | Integer| Serial Data Number (greater than or equal to 1)              |
    |D  17        |    1 | Integer| Evaluation of Individual Character Image (>= 0)              |
    |   18        |    1 | Integer| Evaluation of Character Group (greater than or equal to 0)   |
    |P  19        |    1 | Integer| Male-Female Code ( 1=male, 2=female ) (JIS X 0303)           |
    |a  20        |    1 | Integer| Age of Writer                                                |
    |r  21 -   22 |    2 | Integer| Industry Classification Code (JIS X 0403)                    |
    |t  23 -   24 |    2 | Integer| Occupation Classification Code (JIS X 0404)                  |
    |   25 -   26 |    2 | Integer| Sheet Gatherring Date (19)YYMM                               |
    |   27 -   28 |    2 | Integer| Scanning Date (19)YYMM                                       |
    |   29        |    1 | Integer| Sample Position X on Sheet (greater than or equal to 0)      |
    |   30        |    1 | Integer| Sample Position Y on Sheet (greater than or equal to 0)      |
    |   31 -   60 |   30 |        | (undefined)                                                  |
    |-------------|------|--------|--------------------------------------------------------------|
    |   61 - 8188 | 8128 | Packed | 16 Gray Level (4bit/pixel) Image Data                        |
    |             |      |        | 128(X-axis size) * 127(Y-axis size) = 16256 pixels           |
    |-------------|------|--------|--------------------------------------------------------------|
    | 8189 - 8199 |   11 |        | (uncertain)                                                  |
    --------------------------------------------------------------------------------------------
    '''
    r = f.read(RECORD_SIZE)
    record = struct.unpack('>H2s8sI4B4H2B30x8128s11x', r)
    kana_img = Image.frombytes('F', (128, 127), record[14], 'bit', 4)
    kana_img_greyscale = kana_img.convert('L')

    return record + (kana_img_greyscale,)
  
def read_ETL8G_files():
    '''
    From ETL8 spec:

    Each file contains 5 data sets except ETL8G_33 ([#records] = [#categories] * [#datasets], [#bytes] = [#records] * 8199).
    Each data set contains 956 characters written by a writer.
    Each writer wrote 10 sheets per data set ([#sheets] = 10 * [#data sets])
    '''
    # Uncomment code snippet and place in appropriate spots for testing purposes:
    # kana_to_sound = {}
    ## The second position of the record contains the JIS x 0208 code of the kana
    # jis_0208_code = record[1]
    # kana_to_sound[(b'\033$B' + jis_0208_code).decode('iso2022_jp')] = record[2].decode('utf-8')

    # number of kana = 71, number of people = 160, image_x = 127, image_y = 128
    np_array = np.zeros([71, 160, 127, 128], dtype=np.uint8)

    # There are 33 ETL8 files. Process the first 32
    for file_num in range(1, 33):
        ETL_file = 'ETLfiles/ETL8G/ETL8G_{:02d}'.format(file_num)
        with open(ETL_file, 'rb') as f: 
            # Each file (except for the last one, which we won't use) has 5 datasets.
            # Each dataset has 956 categories and equivelantly 956 records
            for i_dataset in range(NUM_DATASETS):
                for i_category in range(NUM_CATEGORIES):
                    record = read_ETL8G_record(f)
                    # In the ETL8 spec, the third position of the record (i.e. record[2]) contains "JIS typical reading".
                    kana_reading = record[2]

                    # The grayscale image of the kana as we saved in our implementation
                    kana_img = record[-1]
                    
                    # In this format Hiragana characters have the suffix .HIRA with the exception of 'WO' which is in the format 'O.WO.HIR'.
                    # 'KAI' and 'HEI' also has sufix .HIRA but we don't want them.
                    if b'.HIRA' in kana_reading or kana_reading == b'O.WO.HIR':
                        if kana_reading != b'KAI.HIRA' and kana_reading != b'HEI.HIRA':
                            np_array[HIRAGANA_READING_TO_ORDER[kana_reading], (NUM_DATASETS * file_num) -  (5 - i_dataset)] = np.array(kana_img)
    
    np.savez_compressed('hiragana_images.npz', np_array)

if __name__ == '__main__':
    read_ETL8G_files()