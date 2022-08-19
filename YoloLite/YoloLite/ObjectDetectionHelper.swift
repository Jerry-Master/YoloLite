//
//  ObjectDetectionHelper.swift
//  YoloLite
//
//  Created by Jose Perez Cano on 16/8/22.
//


import TensorFlowLite
import UIKit


/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct Result {
  let inferenceTime: Double
  //let detections: [Detection]
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `ObjectDetector`.
class ObjectDetectionHelper: NSObject {
    

  // MARK: Private properties
  /// TensorFlow Lite `ObjectDetector` object for performing object detection using a given model.
  private var detector: Interpreter
  private let names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

  // MARK: - Initialization
  /// A failable initializer for `ObjectDetectionHelper`.
  ///
  /// - Parameter modelFileInfo: The TFLite model to be used.
  /// - Parameter:
  ///   - threadCount: Number of threads to be used.
  ///   - scoreThreshold: Minimum score of objects to be include in the detection result.
  ///   - maxResults: Maximum number of objects to be include in the detection result.
  /// - Returns: A new instance is created if the model is successfully loaded from the app's main
  /// bundle.
  init?(modelFileName: String, modelFileExtension: String) {

    // Construct the path to the model file.
    guard
      let modelPath = Bundle.main.path(
        forResource: modelFileName,
        ofType: modelFileExtension
      )
    else {
      print("Failed to load the model file with name: \(modelFileName).")
      return nil
    }
    
    do {
      // Initialize an interpreter with the model.
      detector = try Interpreter(modelPath: modelPath)
      print("Succesfully loaded interpreter located at: \(modelPath)")
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    super.init()
  }
  /// - Returns: The internal interpreter
  func getInterpreter() -> Interpreter{
    return detector
  }
  
  func flattenTranspose(_ arr: [UInt8]) -> [Float32]{
    var res2 = [Float32](repeating: 0, count: 640*640*3)
    var intCounter = 0
    for k in 0...2{
        for i in 0...639{
            for j in 0...639{
                res2[intCounter] = Float32(arr[640*4*i + 4*j + k]) / Float32(255)
                intCounter += 1
            }
        }
    }
    return res2
  }
  
  func toData(_ img: inout UIImage) -> Data?{
    let pix = img.getPixelValues()!
    let pixFlat = flattenTranspose(pix)
    return Data(copyingBufferOf: pixFlat)
  }
  
  /// Detect objects from the given frame.
  ///
  /// This method handles all data preprocessing and makes calls to run inference on a given frame
  /// through the `Detector`. It then formats the inferences obtained and returns results
  /// for a successful inference.
  ///
  /// - Parameter frame: The target frame.
  /// - Returns: The same frame with the detected objects marked with rectangles.
  func detect(frame: inout UIImage) -> Void {
    let frameSize = frame.size
    let outputTensor: Tensor
    var startTime: Date = Date()
    var resizingTime: TimeInterval = 0
    var preprocessingTime: TimeInterval = 0
    var inferenceTime: TimeInterval = 0
    var postprocessingTime: TimeInterval = 0
    // Resize to 640x640
    var frameAux: UIImage = UIImage(cgImage: (frame.cgImage?.resize(size: CGSize(width: 640, height: 640))!)!)
    var now = Date()
    resizingTime = now.timeIntervalSince(startTime)
    startTime = Date()
    do {
      // Format the pixels into a 1D array
      guard let rgbData = toData(&frameAux)
      else {
        print("Failed to convert the image buffer to RGB data.")
        return
      }
      
      // Calculate preprocessing time.
      now = Date()
      preprocessingTime = now.timeIntervalSince(startTime)
      startTime = Date()

      // Allocate memory for the model's input `Tensor`s.
      try detector.allocateTensors()

      // Copy the RGB data to the input `Tensor`.
      try detector.copy(rgbData, toInputAt: 0)

      // Run inference by invoking the `Interpreter`.
      try detector.invoke()

      // Get the output `Tensor` to process the inference results.
      outputTensor = try detector.output(at: 0)

      // Calculate inference time.
      now = Date()
      inferenceTime = now.timeIntervalSince(startTime)
      startTime = Date()
      
      // Parse the output into a 1D array of bboxes
      let bboxes = outputTensor.data.toArray(type: Float32.self)
      //print(bboxes)
      
      // Paint the bboxes with a score greater than 0.5
      //let xRatio = frameSize.width / 640 , yRatio = frameSize.height / 640
      let xRatio = CGFloat(1), yRatio = CGFloat(1)
      for i in 0...99{
        let score = CGFloat(bboxes[7*i + 6])
        if score < 0.5 {continue}
        let x = CGFloat(bboxes[7*i + 1]) * xRatio, y = CGFloat(bboxes[7*i + 2]) * yRatio, x2 = CGFloat(bboxes[7*i + 3]) * xRatio, y2 = CGFloat(bboxes[7*i + 4]) * yRatio
        let w = x2 - x, h = y2 - y
        let nameIndex = Int(bboxes[7*i + 5])
        frameAux = frameAux.drawRect(at: CGPoint(x: x, y: y), with: CGSize(width: w, height: h), text: names[nameIndex] + ": " + String(format: "%.2f", Double(score)))!
      }
      frame = frameAux.scaledImage(with: frameSize)!
      now = Date()
      postprocessingTime = now.timeIntervalSince(startTime)
      
      // Show the time for each step
      print("Resizing time: \(resizingTime)")
      print("Preprocessing time \(preprocessingTime)")
      print("Inference time \(inferenceTime)")
      print("Postprocessing time \(postprocessingTime)")
      
    } catch let error {
      // Error handling...
      print(error)
    }
  }
}
