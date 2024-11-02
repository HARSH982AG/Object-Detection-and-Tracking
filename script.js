const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true,
        });
        video.srcObject = stream;

        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                resolve(video);
            };
        });
    } catch (error) {
        console.error('Error setting up camera:', error);
    }
}

async function loadModel() {
    try {
        // Load the YOLOv5 model from the TensorFlow.js model repository
        const model = await tf.loadGraphModel('https://tfhub.dev/tensorflow/yolo_v4/1/default/1', { fromTFHub: true });
        return model;
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

async function detectFrame(model) {
    try {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const inputTensor = tf.browser.fromPixels(canvas);
        const predictions = await model.executeAsync(inputTensor.expandDims(0));

        // Process predictions (this part will vary based on the output format of the model)
        const boxes = predictions[0].arraySync();
        const scores = predictions[1].arraySync();
        const classes = predictions[2].arraySync();

        drawPredictions(boxes, scores, classes);
        requestAnimationFrame(() => detectFrame(model));
    } catch (error) {
        console.error('Error detecting frame:', error);
    }
}

function drawPredictions(boxes, scores, classes) {
    try {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        boxes.forEach((box, index) => {
            const [yMin, xMin, yMax, xMax] = box;

            if (scores[index] > 0.5) { // Confidence threshold
                const width = xMax - xMin;
                const height = yMax - yMin;
                ctx.strokeStyle = 'red';
                ctx.lineWidth = 2;
                ctx.strokeRect(xMin * canvas.width, yMin * canvas.height, width * canvas.width, height * canvas.height);
                ctx.fillStyle = 'red';
                ctx.fillText(`Class: ${classes[index]} Score: ${scores[index].toFixed(2)}`, xMin * canvas.width, yMin * canvas.height);
            }
        });
    } catch (error) {
        console.error('Error drawing predictions:', error);
    }
}

async function main() {
    try {
        await setupCamera();
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const model = await loadModel();
        detectFrame(model);
    } catch (error) {
        console.error('Error in main function:', error);
    }
}

main();