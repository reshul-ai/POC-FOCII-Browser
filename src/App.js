import pic from  "./assets/companyLogo.jpg"
import "./App.css"
import React,{useEffect, useRef, useState} from "react"

import '@mediapipe/face_detection'
import '@tensorflow/tfjs-core'
import '@tensorflow/tfjs-backend-webgl'
import * as faceDetection from '@tensorflow-models/face-detection'
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection'
import Webcam from "react-webcam";
import XMLData from './mlmodels/input/facedetection/haarcascade_frontalface_default.xml';
import axios from "axios"
import cv from "@techstark/opencv-js";
import { loadHaarFaceModels, detectHaarFace } from "./faceDetection";


function App() {

  let videoRef = useRef(null)
  let photoRef = useRef(null)
  const webcamRef = useRef(null);

// get access to web camera
  useEffect(()=>{
    // setUpCamera();
    loadHaarFaceModels().then(() => {
      console.log('Face Detection Model loaded successfull');
    }).catch(err => console.error(err));
  },[videoRef])


  let setUpCamera = () =>{

    navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    }).then((stream)=>{
        let video = videoRef.current;
        video.srcObject = stream;
        video.play();
      })
      .catch(err =>{
        console.error(err)
      })


  } 

  // clear image
  const clearImage = () =>{
    let photo = photoRef.current
    let ctx = photo.getContext('2d');
    ctx.clearRect(0,0,650,499);
  }



  const faceDetectionModel = async()=>{
    const model = faceDetection.SupportedModels.MediaPipeFaceDetector;
    const detectorConfig = {
      runtime:'mediapipe',
      solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_detection'
    }
    let detector = await faceDetection.createDetector(model,detectorConfig);
    const estimationConfig = {flipHorizontal: false};
    const faces = await detector.estimateFaces(videoRef.current, estimationConfig);
    console.log(faces);
    const width = 650
    const height = 499 
    let video = videoRef.current
    let photo = photoRef.current
    photo.width = width
    photo.height = height
    let ctx = photo.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0, width, height);
    faces.forEach(pred =>{
      ctx.beginPath();
      ctx.lineWidth = "4";
      ctx.strokeStyle = "red";
      ctx.rect(
        pred.box.xMin,
        pred.box.yMin,
        pred.box.height,
        pred.box.width,
      );
      ctx.stroke();
    }) 
  }

  
  // Detecting through face mesh
  const faceMeshDetection = async() =>{
    console.log(`Inside face Mesh detection`);

    const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
    const detectorConfig = {
      runtime: 'mediapipe', // or 'tfjs'
      solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh',
    }
    const detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
    const estimationConfig = {flipHorizontal: false};

    const faces = await detector.estimateFaces(videoRef.current, estimationConfig);

    console.log(faces)

    const width = 650
    const height = 499 
    let video = videoRef.current
    let photo = photoRef.current
    photo.width = width
    photo.height = height
    let ctx = photo.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0, width, height);

     

    faces[0].keypoints.forEach(prediction =>{
        // Draw Dots

        for (let i = 0; i < faces[0].keypoints.length; i++) {
          //  if(prediction.name == undefined){
            const x = prediction.x;
            const y = prediction.y;

            ctx.beginPath();
            ctx.arc(x, y, 2, 0, 2 * Math.PI);
            ctx.fillStyle = "aqua";
            ctx.fill();
          //  }
        }
    })
  }

  // Triangle drawing method
const drawPath = (ctx, points, closePath) => {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.strokeStyle = "grey";
  ctx.stroke(region);
};



  // face detection with model kept in a file in javascript

  const faceDetectionfromFile = () => {
    getXMLFile();
    // const imageSrc = webcamRef.current.getScreenshot();
    const imageSrc = videoRef.current;
    console.log(imageSrc);
    photoRef.current.src= imageSrc;
    try {
      const img = cv.imread(photoRef.current);
      detectHaarFace(img);
      cv.imshow(photoRef.current, img);
    } catch (error) {
      console.log('Inside error',error);
    }
  }


  // read XML file and call opencv model

  let getXMLFile = () =>  {
    axios.get(XMLData, {
      "Content-Type": "application/xml; charset=utf-8"
   })
   .then((response) => { 
      window.cv = cv;
      // console.log('Your xml file as string', response.data);
      let pixels = cv.imread("C:/Users/BrainAlive/Desktop/BrainAliveTasks/facialfeaturesdetectionmodel/src/mlmodels/input/images/img2.png");
      let classifier = cv.CascadeClassifier(response.data);
      let bboxes = classifier.detectMultiScale(pixels)
      for(let box in bboxes){
        let [x,y,width,height] = box
        let x2 = x + width , y2 = y + height
        cv.rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
      }
      cv.imshow(pixels);
   });

  }

  return (
      <div className='container-fluid'>
        <div className='row pt-3'>
          <div className="col-md-2"></div>
          <div className='col-md-2'>
              <img className="img-fluid" width="120px" src={pic} alt='Brainalive logo'/>
          </div>
          <div className='col'>
              <h1>Welcome to BrainAlive</h1>
          </div>
        </div>
        <div className='row text-center'>
            <div className='col'><hr /></div>
        </div>
        <div className='row text-center'>
            <div className='col'><p>Please keep your face inside the box</p></div>
        </div>
        <div className='row text-center'>
            <div className="col">
              <video className="container" ref={videoRef} height="400px"></video>
            </div>
        </div>
        <div className="row text-center">
           <div className="col pt-2 mb-4">
                {/* <button onClick={faceDetectionfromFile} className="btn btn-danger">Take Image!</button> */}
                <button onClick={faceDetectionfromFile} className="btn btn-danger">Take Image!</button>
                <button onClick={clearImage} className="btn btn-primary">Clear Image!</button>
            </div>
        </div>
        <div className='row text-center'>
            <div className="col">
            <canvas className="container" ref={photoRef} id="canvas" height="200px"></canvas>
            </div>
        </div>
     </div>
  );
}

export default App;
