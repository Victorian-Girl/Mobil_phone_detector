// get access to webcam
let video = document.getElementById('video'); // video element
const canvasElement = document.getElementById('main-canvas'); // canvas element
const canvasCtx = canvasElement.getContext('2d'); // canvas context

// Loading the custom model for classification
var model = undefined;
const URL = "models/Mobile/"                // changer le nom du dossier ici, le nom du dossier est le nom de l'objet que vous voulez détecter

async function loadModel() {
    model = await tf.loadGraphModel(URL + "model.json", URL)
    console.log("Model Loaded");

    startVideo(); // start video
}

loadModel();

function startVideo() {
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })  // get video from webcam

        .then(function (stream) {                    // on success, stream it in video tag
            video.srcObject = stream;               // stream it in video tag
            video.play();                          // play the video
            console.log("Webcam started");         // log to console
        })

        .catch(function (err) {                     // on error, log to console
            console.log(err);                    // log error to console
        });
}



video.addEventListener("loadeddata", () => {

    // changer le nom de la classe ici, le nom de la classe est le nom de l'objet que vous voulez détecter
    // si plusieurs classes, ajouter une ligne avec le numéro de la classe et le nom de la classe

    var className = {
        1: {
            name: 'mobile', id: 1
        },
    }

    setInterval(async () => {

        // Get image from webcam
        const img = tf.browser.fromPixels(video)
        const resized = tf.image.resizeBilinear(img, [960, 720])
        const casted = resized.cast('int32')
        const expanded = casted.expandDims(0)
        const obj = await model.executeAsync(expanded)
        // console.log(obj)

        // ici changer les valeurs de obj pour avoir les valeurs de chaque classe en testant avec un console.log(obj[0].dataSync())
        const scores = obj[5].arraySync()
        const classes = obj[3].dataSync()
        const boxes = obj[0].dataSync()
        // console.log(obj[8].arraySync())

        // eliminated  1, 2, 4, 6, 7, 8

        detections = buildDetectedObjects(scores, 0.6, 960, 720, boxes, classes, className)
        console.log(detections)



        // Draw
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height)
        canvasCtx.drawImage(video, 0, 0, 960, 720)

        if (detections.length > 0) {

            detections.forEach((detection, i) => {
                bbox = detection.bbox
                // Draw
                canvasCtx.beginPath()
                canvasCtx.fillStyle = "red";          // color of the box is red
                canvasCtx.fillRect(bbox[0] - 2, bbox[1] - 40, bbox[2], 40)
                canvasCtx.lineWidth = 10;             // width of the box is 10
                canvasCtx.fillStyle = "white";        // color of the text is white
                canvasCtx.font = "30px Arial";        // font of the text is Arial
                canvasCtx.fillText(detection.label + " " + parseInt(detection.score * 100) + "%", bbox[0] + 10, bbox[1] - 10)
                canvasCtx.lineWidth = 4;             // width of the box is 10
                canvasCtx.strokeStyle = "red";          // color of the box is red
                canvasCtx.rect(bbox[0], bbox[1], bbox[2], bbox[3])


                canvasCtx.stroke()
            })

        }




        tf.dispose(img)
        tf.dispose(resized)
        tf.dispose(casted)
        tf.dispose(expanded)
        tf.dispose(obj)


    }, 10)


})

function buildDetectedObjects(scores, threshold, imageWidth, imageHeight, boxes, classes, classNames) {
    const detectionObjects = []

    scores[0].forEach((score, i) => {
        currentScore = score[classes[i]]
        if (currentScore > threshold) {
            const bbox = [];
            const minY = boxes[i * 4] * imageHeight;
            const minX = boxes[i * 4 + 1] * imageWidth;
            const maxY = boxes[i * 4 + 2] * imageHeight;
            const maxX = boxes[i * 4 + 3] * imageWidth;
            bbox[0] = minX;
            bbox[1] = minY;
            bbox[2] = Math.abs(maxX - minX);
            bbox[3] = maxY - minY;
            detectionObjects.push({
                class: classes[i],
                label: classNames[classes[i]].name,
                score: currentScore,
                bbox: bbox
            })
        }
    })

    return detectionObjects
}