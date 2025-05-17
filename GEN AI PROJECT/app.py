from flask import Flask, request, render_template
from PIL import Image
import base64
from io import BytesIO
from transformers import pipeline

app = Flask(__name__)

# Load the background removal model with trusted custom code
bg_removal = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result_img_data = None
    input_img_data = None
    error = None

    if request.method == 'POST':
        if 'image' not in request.files:
            error = "No image uploaded"
        else:
            try:
                file = request.files['image']
                image = Image.open(file.stream).convert("RGB")

                # Encode original image
                input_buf = BytesIO()
                image.save(input_buf, format='PNG')
                input_img_data = base64.b64encode(input_buf.getvalue()).decode('utf-8')

                # Remove background (FIXED LINE)
                result = bg_removal(image)

                # Encode result image
                output_buf = BytesIO()
                result.save(output_buf, format='PNG')
                result_img_data = base64.b64encode(output_buf.getvalue()).decode('utf-8')

            except Exception as e:
                error = f"Error: {str(e)}"

    return render_template('index.html',
                           input_img_data=input_img_data,
                           result_img_data=result_img_data,
                           error=error)

if __name__ == '__main__':
    app.run(debug=True)
