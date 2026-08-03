from flask import Flask, Response, jsonify, render_template_string
import cv2
import easyocr
import numpy as np
import base64

app = Flask(__name__)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
cap.set(cv2.CAP_PROP_AUTOFOCUS,1)

reader = easyocr.Reader(['en'], gpu=False)
latest = None

HTML="""
<!doctype html>
<html>
<head>
<title>EasyOCR Test</title>
<style>
body{font-family:Arial;background:#222;color:#fff;text-align:center}
img{border:2px solid #555}
button{padding:12px 25px;font-size:18px;margin:15px}
pre{background:#111;padding:10px;display:inline-block;text-align:left}
</style>
</head>
<body>
<h2>EasyOCR Camera Test</h2>
<img src="/video_feed" width="800"><br>
<button onclick="proc()">Process Image</button>
<div id="out"></div>
<script>
async function proc(){
 let r=await fetch('/process',{method:'POST'});
 let d=await r.json();
 document.getElementById('out').innerHTML=
 `<h3>Rotation: ${d.rotation}</h3>
 <h3>Confidence: ${d.confidence.toFixed(3)}</h3>
 <img width="500" src="data:image/jpeg;base64,${d.image}"><br>
 <pre>${d.text}</pre>`;
}
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

def gen():
    global latest
    while True:
        ok,frame=cap.read()
        if not ok: continue
        latest=frame.copy()
        _,buf=cv2.imencode(".jpg",frame)
        yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buf.tobytes()+b'\r\n')

@app.route("/video_feed")
def video():
    return Response(gen(),mimetype="multipart/x-mixed-replace; boundary=frame")

def preprocess(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(2.0,(8,8))
    g=clahe.apply(g)
    g=cv2.GaussianBlur(g,(0,0),2)
    g=cv2.addWeighted(clahe.apply(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)),1.5,g,-0.5,0)
    return cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)

@app.route("/process",methods=["POST"])
def process():
    global latest
    if latest is None:
        return jsonify(error="No frame")
    frame=latest.copy()
    rots=[
      ("0",frame),
      ("90",cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)),
      ("180",cv2.rotate(frame,cv2.ROTATE_180)),
      ("270",cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE))
    ]
    best_txt=""
    best_conf=-1
    best_rot=""
    best_img=frame
    for name,img in rots:
        p=preprocess(img)
        res=reader.readtext(p)
        if not res: continue
        conf=np.mean([x[2] for x in res])
        if conf>best_conf:
            best_conf=conf
            best_rot=name
            best_txt="\n".join([x[1] for x in res])
            best_img=img.copy()
            for box,text,c in res:
                pts=np.array(box,dtype=int)
                cv2.polylines(best_img,[pts],True,(0,255,0),2)
                cv2.putText(best_img,text,(pts[0][0],pts[0][1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    _,buf=cv2.imencode(".jpg",best_img)
    return jsonify(
        rotation=best_rot,
        confidence=float(best_conf if best_conf!=-1 else 0),
        text=best_txt,
        image=base64.b64encode(buf).decode()
    )

if __name__=="__main__":
    app.run(debug=True)