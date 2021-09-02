const  modelPath = './model_mn64_js/model.json';
const faceCascadeFile = "./haarcascade_frontalface_default.xml"; 

const maskBoxColor =  [0, 255, 0, 255];
const noMaskBoxColor =  [255, 0, 0, 255];
const FPS = 60; // maximum fps 

let streaming = false;

// load calssifier
let classifier = new cv.CascadeClassifier();

let utils = new Utils('errorMessage'); //use utils class
let model  = null;

let capbutton = document.getElementById("capbutton");
let fpsText = document.getElementById("fps");
let video = document.getElementById("videoInput"); // video is the id of video 

navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then(function(stream) {
        video.srcObject = stream;
        // video.play();
        // load fascade file
        utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
            console.log( 'cascade loaded: ', classifier.load(faceCascadeFile), faceCascadeFile); // in the callback, load the cascade from file 
            loadModel();
        });
    })
    .catch(function(err) {
        console.log("An error occurred! " + err);
    });

let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
let gray = new cv.Mat();
let face_im = new cv.Mat();
let faces = new cv.RectVector();

// get the camera
let cap = new cv.VideoCapture(video);

// hide video input canvas
document.getElementById("videoInput").style.visibility = "hidden";
capbutton.style.visibility = "hidden";
fpsText.style.visibility = "hidden";

const faceImWidth = 64;
const faceImSize = new cv.Size( faceImWidth, faceImWidth );

// load the tf model
async function loadModel() {

    model = await tf.loadLayersModel( modelPath );

    if (model != null) {
        console.log ( 'model loaded...' );
        capbutton.style.visibility = "visible";
    }
    
}




// process video funtion for the opencv 
function processVideo() {
    try {
        if (!streaming) {
            return;
        }

        let begin = Date.now();

        // start processing.
        cap.read(src);
       
        // get gray color image
        cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY, 0);

        src.copyTo( dst );

        // detect faces
        classifier.detectMultiScale( gray, faces, 1.05, 5); // 1.05 
        
        // go though each face 
        for (let i = 0; i < faces.size(); i++) {

            let face = faces.get(i);
            let point1 = new cv.Point( face.x, face.y );
            let point2 = new cv.Point( face.x + face.width, face.y + face.height );

            let sz = Math.max( face.width, face.height );

            face_im = src.roi( new cv.Rect(face.x, face.y, sz, sz) );
            
            // You can try more different parameters
            cv.resize(face_im, face_im,  faceImSize, 0, 0, cv.INTER_AREA);
            

            cv.cvtColor( face_im, face_im, cv.COLOR_RGBA2RGB, 0);

            // predict for the face
            let predict =  model.predict( tf.tensor(face_im.data, [1, faceImWidth, faceImWidth, 3]).mul(tf.scalar( 1.0 / 255.0 )));
            let predictArray = predict.arraySync();

            let boxColor = noMaskBoxColor;
            let text = "No Mask";

            if (predictArray[0][0] >= predictArray[0][1]) {
                boxColor = maskBoxColor;
                text = "mask";
            } 
            
            cv.rectangle(dst, point1, point2, boxColor, 2);
            cv.rectangle(dst, new cv.Point( face.x - 1, face.y ), new cv.Point( face.x + face.width + 1, face.y - 20), boxColor, -1);
            cv.putText( dst, text, {x:face.x + 2, y:face.y - 2}, cv.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255, 255] );
        }

    
        cv.imshow( "canvasOutput", dst );
     
        let delay = 1000 / FPS - (Date.now() - begin);
        fpsText.innerHTML = "FPS: "  + Math.round(1000 / (Date.now() - begin))
        setTimeout(processVideo, delay);
        // requestAnimationFrame(processVideo);

    } catch (err) {;
        console.log( "Error: " + err );
    }
}
    

function start() {
    if (!streaming) {
        console.log( "Streming starting..." );
        video.play();
        streaming = true;
        capbutton.innerHTML= "Stop Capturing";
        fpsText.style.visibility = "visible"
        setTimeout(processVideo, 0);   
    } else {
        streaming = false;
        capbutton.innerHTML = "Start Capturing"
    }
}


function clear() {
    src.delete();
    dst.delete();
    faces.delete();
    gray.delete();
    classifier.delete();
    face_im.delete();
    console.log("clear...");
}



