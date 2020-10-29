from flask import render_template
from flask import Flask, request, json
from utils.functions import save_location
from utils.functions import load_location
from utils.functions import decode_image
from utils.functions import crop_image
from utils.functions import call_tf_serving

app = Flask(__name__)


@app.route("/")
@app.route("/define_form")
def define_form():
    return render_template("define_form.html")


@app.route("/test")
def test():
    return render_template("test.html")


@app.route("/api/submit", methods=["POST"])
def submit_form():
    location = request.form.get("location")
    form_id = request.form.get("form_id")
    save_location(location, form_id)
    print("Submited", form_id, location)
    return "0"


@app.route("/api/ocr", methods=["POST"])
def get_text_ocr():
    image_base64 = request.files.get("image")
    form_id = request.form.get("form_id")
    location = load_location(form_id)
    if location is None:
        # return ket qua k co gi
        print("error")
        # return '0'
    else:
        image = decode_image(image_base64)
        cropped_list = crop_image(image, location)
        text = call_tf_serving(cropped_list)
        print(text)
        return json.dumps({"text": text})


if __name__ == "__main__":
    app.run(debug=True)
