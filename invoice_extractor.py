from groq import Groq
import base64
import os

from dotenv import load_dotenv

import json


class InvoiceExtractor:
    def __init__(self):
        load_dotenv()
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


    @staticmethod
    def encode_image(image_path):
      with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


    @staticmethod
    def read_file(file_path):
        with open(file_path, 'r') as file:
            return file.read()



    def extract_info_from_receipt(self, image_path):
        base64_image = InvoiceExtractor.encode_image(image_path)

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": InvoiceExtractor.read_file("./prompt.txt")},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            response_format={"type": "json_object"},
            temperature=0
        )

        return json.loads(chat_completion.choices[0].message.content)


if __name__ == "__main__":
    # Path to your image
    image_path = "./dataset/receipts/1000-receipt.jpg"
    invoice_extractor = InvoiceExtractor()
    print(invoice_extractor.extract_info_from_receipt(image_path))

    
