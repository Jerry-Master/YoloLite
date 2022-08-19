//
//  ImagePicker.swift
//  YoloLite
//
//  Created by Jose Perez Cano on 15/8/22.
//

import SwiftUI
import TensorFlowLite

struct ImagePicker: UIViewControllerRepresentable {
    @Environment(\.presentationMode) private var presentationMode
    var sourceType: UIImagePickerController.SourceType = .photoLibrary
    @Binding var selectedImage: UIImage
    @Binding var rat: Double
    @Binding var interp: ObjectDetectionHelper?

    func makeUIViewController(context: UIViewControllerRepresentableContext<ImagePicker>) -> UIImagePickerController {

        let imagePicker = UIImagePickerController()
        imagePicker.allowsEditing = false
        imagePicker.sourceType = sourceType
        imagePicker.delegate = context.coordinator

        return imagePicker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: UIViewControllerRepresentableContext<ImagePicker>) {

    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

        var parent: ImagePicker

        init(_ parent: ImagePicker) {
            self.parent = parent
        }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {

            if let image = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {
                let colorSpace = image.cgImage?.colorSpace
                let validRGB = colorSpace == CGColorSpace(name: CGColorSpace.sRGB) && colorSpace != nil
                if !validRGB {
                    parent.selectedImage = image.toRGB()!
                }
                else {
                    parent.selectedImage = image
                }
                if !(image.size.height / image.size.width).isNaN{
                    parent.rat = image.size.height / image.size.width
                }
            }

            parent.presentationMode.wrappedValue.dismiss()
        }

    }
}
