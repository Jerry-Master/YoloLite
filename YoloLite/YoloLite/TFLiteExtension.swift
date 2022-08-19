// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import CoreGraphics
import Foundation
import UIKit

// MARK: - CGImage

/// Extension of iOS classes that is useful for working with TensorFlow Lite computer vision models.
extension CGImage {
    /// Creates and returns a new image with a different size.
    ///
    /// - Parameter size: The size to scale the image to.
    /// - Returns: The scaled image or `nil` if image could not be modified.
    func resize(size:CGSize) -> CGImage? {
        let width: Int = Int(size.width)
        let height: Int = Int(size.height)

        let bytesPerPixel = self.bitsPerPixel / self.bitsPerComponent
        let destBytesPerRow = width * bytesPerPixel


        guard let colorSpace = self.colorSpace else { return nil }
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: self.bitsPerComponent, bytesPerRow: destBytesPerRow, space: colorSpace, bitmapInfo: self.alphaInfo.rawValue) else { return nil }

        context.interpolationQuality = .high
        context.draw(self, in: CGRect(x: 0, y: 0, width: width, height: height))

        return context.makeImage()
    }
}

// MARK: - UIImage

/// Extension of iOS classes that is useful for working with TensorFlow Lite computer vision models.
extension UIImage {
    
    /// Creates and returns the image in RGB color space
    ///
    /// - Parameter size: The size to scale the image to.
    /// - Returns: The new image or `nil` if image could not be modified.
    func toRGB() -> UIImage? {
        let imageSize = self.size
        let scale: CGFloat = 0
        UIGraphicsBeginImageContextWithOptions(imageSize, false, scale)
        defer { UIGraphicsEndImageContext() }
        self.draw(at: CGPoint.zero)
        return UIGraphicsGetImageFromCurrentImageContext()
    }
    
    /// Draws a rectangle with text in the given position.
    /// - Parameters:
    ///  - at: The top-left corner of the rectangle.
    ///  - with: The width and height of the rectangle.
    ///  - text: String with the text to put.
    /// - Returns: The new image with the rectangle or `nil` if image could not be modified.
    func drawRect(at: CGPoint, with rect: CGSize, text: String) -> UIImage?{
        let imageSize = self.size
        let scale: CGFloat = 0
        
        // Create context
        UIGraphicsBeginImageContextWithOptions(imageSize, false, scale)
        defer { UIGraphicsEndImageContext() }
        self.draw(at: CGPoint.zero)
        
        // Draw rectangle
        let context = UIGraphicsGetCurrentContext()
        context!.setStrokeColor(UIColor.purple.cgColor)
        context!.setLineWidth(5)
        let rectangle = CGRect(x: at.x, y: at.y, width: rect.width, height: rect.height)
        context!.stroke(rectangle)
        
        // Text attributes
        let font=UIFont(name: "Helvetica-Bold", size: 28)!
        let textStyle=NSMutableParagraphStyle()
        textStyle.alignment=NSTextAlignment.center
        let textColor=UIColor.white
        let backgroundColor=UIColor.purple
        let attributes=[NSAttributedString.Key.font:font, NSAttributedString.Key.paragraphStyle:textStyle, NSAttributedString.Key.foregroundColor:textColor, NSAttributedString.Key.backgroundColor:backgroundColor]
        
        // Draw text
        let br = CGPoint(x: at.x, y: at.y + rect.height)
        text.draw(at: br, withAttributes: attributes)
        
        return UIGraphicsGetImageFromCurrentImageContext()
    }

    /// Creates and returns a new image with a different size.
    ///
    /// - Parameter size: The size to scale the image to.
    /// - Returns: The scaled image or `nil` if image could not be modified.
    func scaledImage(with size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        defer { UIGraphicsEndImageContext() }
        self.draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
    
    /// Creates and returns the pixels of the image as a flattened array, in RGBA format.
    ///
    /// - Returns: The pixels array.
    func getPixelValues() -> [UInt8]? {
            let size = self.size
            var width = 0
            var height = 0
            
            width = Int(size.width)
            height = Int(size.height)
            let bitsPerComponent = 8
            let bytesPerRow = width * 4
            let totalBytes = height * bytesPerRow
            let bitmapInfo = self.cgImage?.bitmapInfo

            let colorSpace = CGColorSpaceCreateDeviceRGB()
            var pixelValues = [UInt8](repeating: 0, count: totalBytes)

            let contextRef = CGContext(data: &pixelValues,
                                      width: width,
                                     height: height,
                           bitsPerComponent: bitsPerComponent,
                                bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                 bitmapInfo: bitmapInfo!.rawValue)
            contextRef?.draw(self.cgImage!, in: CGRect(x: 0.0, y: 0.0, width: CGFloat(width), height: CGFloat(height)))
            return pixelValues
    }  
}

// MARK: - Data
extension Data {
    /// Creates a new buffer by copying the buffer pointer of the given array.
    ///
    /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
    ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
    ///     data from the resulting buffer has undefined behavior.
    /// - Parameter array: An array with elements of type `T`.
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBufferPointer(Data.init)
    }

    /// Convert a Data instance to Array representation.
    func toArray<T>(type: T.Type) -> [T] where T: ExpressibleByIntegerLiteral {
        var array = [T](repeating: 0, count: self.count/MemoryLayout<T>.stride)
        _ = array.withUnsafeMutableBytes { copyBytes(to: $0) }
        return array
    }
}

// MARK: - Constants
private enum Constant {
    static let jpegCompressionQuality: CGFloat = 0.8
    static let alphaComponent = (baseOffset: 4, moduloRemainder: 3)
    static let maxRGBValue: Float32 = 255.0
}
