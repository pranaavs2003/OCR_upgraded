import cv2
import typing
import numpy as np
import os
from detector import detector
from pydantic import BaseModel

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageDetail(BaseModel):
    image_url: str

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("202301111911/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    df = pd.read_csv("202301111911/val.csv").values.tolist()


    try:
        folder_path = './mydata'
        image_url = input('Enter Image URL: ')
        # image_url = "https://res.cloudinary.com/dbzzj25vc/image/upload/v1691164661/checkd/cheques/d2seilqujdawwkjn4sga.jpg"
        image_details = ImageDetail(image_url= image_url)
        
        for entry in os.listdir(folder_path):
            full_path = os.path.join(folder_path, entry)

            if os.path.isfile(full_path):
                img_num = 'mydata/' + entry
                image = cv2.imread(img_num)
                prediction_text = model.predict(image)
        
        
        ocrData = detector(image_details)
        print(ocrData)
                
            
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


    # img_num = 'mydata/' + input('Enter file name: ')
    # print(img_num)
    # image = cv2.imread(img_num)
    # prediction_text = model.predict(image)
    # print('🆓', prediction_text)

    # img_num = int(input('Enter the image number: '))
    # image = cv2.imread('mydata/pic'+str(img_num)+'.jpeg')
    # prediction_text = model.predict(image)
    # print('🆓', prediction_text)
    
    # resize by 4x
    # image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # accum_cer = []
    # for image_path, label in tqdm(df):
    #     image = cv2.imread(image_path)
        

    #     prediction_text = model.predict(image)

    #     cer = get_cer(prediction_text, label)
    #     print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

    #     accum_cer.append(cer)

    #     # resize by 4x
    #     image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
    #     cv2.imshow("Image", image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     break
    

    # print(f"Average CER: {np.average(accum_cer)}")