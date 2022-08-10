import cv from "@techstark/opencv-js";
import { loadData } from "./loadData";

const msize = new cv.Size(0, 0);
let faceCascade;

export async function loadHaarFaceModels() {
  console.log("=======start downloading Haar-cascade models=======");
  return loadData(
    "haarcascade_frontalface_default.xml",
    "mlmodels/input/facedetection/haarcascade_frontalface_default.xml"
  )
    .then(
      () =>
        new Promise((resolve) => {
          setTimeout(() => {
            // load pre-trained classifiers
            faceCascade = new cv.CascadeClassifier();
            faceCascade.load("haarcascade_frontalface_default.xml");
            // resolve();
          }, 2000);
        })
    )
    .then(() => {
      console.log("=======downloaded Haar-cascade models=======");
    })
    .catch((error) => {
      console.error(error);
    });
}

/**
 * Detect faces from the input image.
 * See https://docs.opencv.org/master/d2/d99/tutorial_js_face_detection.html
 * @param {cv.Mat} img Input image
 * @returns the modified image with detected faces drawn on it.
 */
export function detectHaarFace(img) {
  // const newImg = img.clone();
  const newImg = img;

  const gray = new cv.Mat();
  cv.cvtColor(newImg, gray, cv.COLOR_RGBA2GRAY, 0);

  const faces = new cv.RectVector();

  // detect faces
  faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);
  for (let i = 0; i < faces.size(); ++i) {
    const point1 = new cv.Point(faces.get(i).x, faces.get(i).y);
    const point2 = new cv.Point(
      faces.get(i).x + faces.get(i).width,
      faces.get(i).y + faces.get(i).height
    );
    cv.rectangle(newImg, point1, point2, [255, 0, 0, 255]);
  }

  gray.delete();
  faces.delete();

  return newImg;
}
 