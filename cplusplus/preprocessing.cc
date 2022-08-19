#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <time.h>
using namespace std;
using namespace cv;

#define readv(v) for(auto& var : v) cin >> var;
#define read(var) int var; cin >> var;
#define printv(v) for (int var = 0; var < v.size(); var++) cout << (var==0 ? "" : " ") << v[var]; cout << endl;
#define queries(var) int var; cin >> var; while (var--) 
#define use_fast ios::sync_with_stdio(false), cin.tie(0), cout.tie(0)

using ll = long long int;
using vi = vector<ll>;
using vvi = vector<vi>;

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void flattenTranspose(const vector<uchar> & arr, vector<float> & out){
    out = vector<float>(640*640*3, 0);
    int intCounter = 0;
    for (int k = 0; k < 3; k++){
      for (int i = 0; i < 640; i++){
          for (int j = 0; j < 640; j++){
                out[intCounter] = ((float) arr[i*640*3 + j*3 + k]) / 255.0;
                intCounter += 1;
          }
      }
    }
  }

int main(){
    clock_t timer0, timer1;
    timer0 = clock();
    Mat image = imread("../horses.jpg");
    timer1 = clock();
    cout << "Image loading time: " << ((float)(timer1-timer0)) / CLOCKS_PER_SEC << endl;
    cout << type2str(image.type()) << endl;
    // Check for failure
    if (image.empty()){
        cout << "Could not open or find the image" << endl;
        cin.get(); //wait for any key press
        return -1;
    }
    timer0 = clock();
    resize(image, image, Size(640,640));
    timer1 = clock();
    cout << "Image resizing time: " << ((float)(timer1-timer0)) / CLOCKS_PER_SEC << endl;

    std::vector<uchar> array;
    if (image.isContinuous()) {
        timer0 = clock();
        array.assign(image.data, image.data + image.total()*image.channels());
        vector<float> out;
        flattenTranspose(array, out);
        timer1 = clock();
        cout << "Image transposing time: " << ((float)(timer1-timer0)) / CLOCKS_PER_SEC << endl;
    }

    String windowName = "Horses"; //Name of the window
    namedWindow(windowName); // Create a window
    imshow(windowName, image); // Show our image inside the created window.
    waitKey(0); // Wait for any keystroke in the window
    destroyWindow(windowName); //destroy the created window

}
