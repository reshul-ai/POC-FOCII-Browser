// Copyright 2020 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const drawing_utils_mediapipe = () =>{

   let _PRESENCE_THRESHOLD = 0.5,
    _VISIBILITY_THRESHOLD = 0.5,
    _RGB_CHANNELS = 3

    function _normalized_to_pixel_coordinates(
        normalized_x, normalized_y, image_width,
        image_height){
            // """Converts normalized value pair to pixel coordinates."""

            // # Checks if the float value is between 0 and 1.
            function is_valid_normalized_value(value){
                return ((value > 0 || math.isclose(0, value)) && (value < 1 || math.isclose(1, value)))
            } 

            if (!(is_valid_normalized_value(normalized_x) &&
                    is_valid_normalized_value(normalized_y)))
                // # TODO: Draw coordinates even if it's outside of the image bounds.
                return None
            let x_px = min(math.floor(normalized_x * image_width), image_width - 1)
            let y_px = min(math.floor(normalized_y * image_height), image_height - 1)
            return x_px, y_px
        }


    function landmarks_xy_idx(
        image,
        landmark_list){

            // """Draws the landmarks and the connections on the image.

            // Args:
            //     image: A three channel RGB image represented as numpy ndarray.
            //     landmark_list: A normalized landmark list proto message to be annotated on
            //     the image.
                
            // Raises:
            //     ValueError: If one of the followings:
            //     a) If the input image is not three channel RGB.
            //     b) If any connetions contain invalid landmark index.
            // """

            if (!landmark_list)
                return
            if (image.shape[2] != _RGB_CHANNELS)
                // raise ValueError('Input image must contain three channel rgb data.')
                console.log(`Input image must contain three channel rgb data.`)
            let image_rows, image_cols, _ = image.shape
            idx_to_coordinates = {}
            for(const [idx, landmark] of landmark_list.landmark.entries()){
                        // for idx, landmark in enumerate(landmark_list.landmark):
                if ((landmark.HasField('visibility') &&
                    landmark.visibility < _VISIBILITY_THRESHOLD) ||
                    (landmark.HasField('presence') &&
                    landmark.presence < _PRESENCE_THRESHOLD))
                continue
                landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                            image_cols, image_rows)
                if (landmark_px)
                idx_to_coordinates[idx] = landmark_px
            }
            
                
            return idx_to_coordinates     



        }



    function detection_xy_idx(image,detection){

            //         """Draws the detction bounding box and keypoints on the image.

            //   Args:
            //     image: A three channel RGB image represented as numpy ndarray.
            //     detection: A detection proto message to be annotated on the image.
            //     keypoint_drawing_spec: A DrawingSpec object that specifies the keypoints'
            //       drawing settings such as color, line thickness, and circle radius.
            //     bbox_drawing_spec: A DrawingSpec object that specifies the bounding box's
            //       drawing settings such as color and line thickness.

            //   Raises:
            //     ValueError: If one of the followings:
            //       a) If the input image is not three channel RGB.
            //       b) If the location data is not relative data.
            //   """

        if (!detection.location_data)
            return
        
        if (image.shape[2] != _RGB_CHANNELS)
            // raise ValueError('Input image must contain three channel rgb data.')
            console.log(`Input image must contain RGB Data`);
        let image_rows, image_cols, _ = image.shape

        let location = detection.location_data
        if (location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX){
            // raise ValueError(
            //     'LocationData must be relative for this drawing funtion to work.')
            console.log(`Location must be relative for this drawing function to work`)
        }
        
        // # Draws keypoints.
        idx_to_coordinates = {}
        idx = 0
        for( keypoint in location.relative_keypoints){
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,image_cols, image_rows)

            if (keypoint_px)
            idx_to_coordinates[idx] = keypoint_px

            idx = idx+1
        }
        

        return idx_to_coordinates 



    }


}



export default drawing_utils_mediapipe;