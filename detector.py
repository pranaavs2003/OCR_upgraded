from pydantic import BaseModel
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv
import time
import os


load_dotenv()
API_KEY = os.getenv('API_KEY')
ENDPOINT = os.getenv('ENDPOINT')
cv_client = ComputerVisionClient(ENDPOINT,CognitiveServicesCredentials(API_KEY))

class ImageDetail(BaseModel):
    image_url: str
    
def detector(transactionDetails: ImageDetail):
    local_file = transactionDetails.image_url
    response = cv_client.read(local_file,  raw=True)
    operationLocation = response.headers['Operation-Location']
    operation_id = operationLocation.split('/')[-1]
    time.sleep(5)
    result = cv_client.get_read_result(operation_id)
    
    lst_1 = []
    
    if result.status == OperationStatusCodes.succeeded:
        read_results = result.analyze_result.read_results
        for analyzed_result in read_results:
            for line in analyzed_result.lines:
                lst_1.append(line.text)
                print('🆙',line.text)
        
    lst_1 = [item for item in lst_1 if all(char.isascii() for char in item)]
    print(lst_1)
    return lst_1
       
    