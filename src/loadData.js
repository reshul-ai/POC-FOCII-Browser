import cv from "@techstark/opencv-js";

export async function loadData(filePath,url){
     // see https://docs.opencv.org/master/utils.js
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    const data = new Uint8Array(buffer);
    cv.FS_createDataFile("/", filePath, data, true, false, false);
}