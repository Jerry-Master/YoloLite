//
//  ContentView.swift
//  YoloLite
//
//  Created by Jose Perez Cano on 14/8/22.
//

import SwiftUI
import TensorFlowLite
import CoreGraphics

private var horsesPath = Bundle.main.path(forResource: "horses", ofType: "jpg")
private var initImage = UIImage.init(contentsOfFile: horsesPath!)!

struct ContentView: View {
    
    @State private var interp: ObjectDetectionHelper? = ObjectDetectionHelper(
        modelFileName: "yolo",
        modelFileExtension: "tflite"
      )
    @State private var image = initImage
    @State private var showSheet = false
    @State private var useCamera = false
    @State private var aspectRat = Double(initImage.size.height / initImage.size.width)
    
    
    var body: some View {
        VStack {
            Button{
                interp!.detect(frame: &image)
            } label: {
                Image("scope")
                    .renderingMode(.original)
                    .font(.largeTitle)
                    .frame(width: 80, height: 80)
            }
            Spacer()
            Image(uiImage: self.image)
                  .resizable()
                  .aspectRatio(1 / aspectRat, contentMode: .fit)
                  .frame(width: 350)
                  .padding()
                  .sheet(isPresented: $showSheet) {
                            // Pick an image from the photo library:
                      if !useCamera {
                          ImagePicker(sourceType: .photoLibrary, selectedImage: self.$image, rat: self.$aspectRat, interp: $interp)
                      } else {
                          ImagePicker(sourceType: .camera, selectedImage: self.$image, rat: self.$aspectRat, interp: $interp)
                      }

                            //  If you wish to take a photo from camera instead:
                            // ImagePicker(sourceType: .camera, selectedImage: self.$image)
                    }
            
            Spacer()
            HStack {
                Button {
                    useCamera = false
                    showSheet = true
                } label: {
                    Image ("plus.viewfinder")
                        .renderingMode(.original)
                        .font(.largeTitle)
                        .frame(width: 80, height: 80)
                }
                Button {
                    useCamera = true
                    showSheet = true
                } label: {
                    Image ("camera")
                        .renderingMode(.original)
                        .font(.largeTitle)
                        .frame(width: 80, height: 80)
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        Group {
            ContentView()
        }
            
    }
}
