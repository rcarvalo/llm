import pandas as pd
import os
from pdf2image import convert_from_path
import cv2
from paddleocr import PPStructure,save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx

def concatenate(csv_path):

    step = 0
    for filename in os.listdir(csv_path):     
        df = pd.read_csv(csv_path + '/'+filename)
        # Clean up the DataFrame by removing newlines.
        df = df.replace('\n', ' ', regex=True)

        # Reformat the DataFrame to be more readable by language models and save it to a text file.
        with open("./processed_data/table_data"+str(step)+".txt", "w") as f:
            for index, row in df.iterrows():
                row_str = ""
                for col in df.columns:
                    row_str += f"{col}: {row[col]}, "
                formatted = row_str[:-2]
                print(formatted)
                f.write(formatted + "\n")
        step = step +1

def convert_pdf_to_img(pdf_file):

    # Store Pdf with convert_from_path function
    images = convert_from_path(pdf_file)

    for i in range(len(images)):

        # Save pages as images in the pdf
        images[i].save('./images/page'+ str(i) +'.png', 'PNG')

def extracting_table_text_from_image(image_path):

    # English image
    table_engine = PPStructure(recovery=True, lang='en')

    save_folder = './images/output'

    for image in os.listdir(image_path):
        #img_path = './page7.png'
        img = cv2.imread(img_path+'/'+image)
        result = table_engine(img)
        save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

        for line in result:
            line.pop('img')
            print(line)

        h, w, _ = img.shape
        res = sorted_layout_boxes(result, w)
        convert_info_docx(img, res, save_folder, os.path.basename(img_path).split('.')[0])

convert_pdf_to_img('./data/BART.pdf')   
extracting_table_text_from_image('./images')     
concatenate("./csv")

'''
from concatenate
Unnamed: 0: BART, SQuAD 1.1 EM/F1: 88.8/94.6, SQuAD 2.0 EM/F1: 86.1/89.2, MNLI m/mm: 89.9/90.1, SST Acc: 96.6, QQP Acc: 92.5, QNLI Acc: 94.9, STS-B Acc: 91.2, RTE Acc: 87.0, MRPC Acc: 90.4, CoLA Mcc: 62.8

'''
'''
from CSVLoader
'Unnamed: 0: BART\nSQuAD 1.1 EM/F1: 88.8/94.6\nSQuAD 2.0 EM/F1: 86.1/89.2\nMNLI m/mm: 89.9/90.1\nSST Acc: 96.6\nQQP Acc: 92.5\nQNLI Acc: 94.9\nSTS-B Acc: 91.2\nRTE Acc: 87.0\nMRPC Acc: 90.4\nCoLA Mcc: 62.8'

'''