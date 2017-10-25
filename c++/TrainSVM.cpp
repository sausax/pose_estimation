#include "json.hpp"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <dlib/svm.h>


using namespace cv;
using namespace cv::ml;
using namespace std;
using json = nlohmann::json;
//using namespace dlib;

#define VECTOR_SIZE 1100

typedef dlib::matrix<float, VECTOR_SIZE, 1> input_vector;
typedef dlib::linear_kernel<input_vector> kernel_type;


Ptr<SVM> initialize_svm(){
    Ptr< SVM > svm = SVM::create();
    /* Default values to train SVM */
    svm->setCoef0( 0.0 );
    svm->setDegree( 3 );
    svm->setTermCriteria( TermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3 ) );
    svm->setGamma( 0 );
    svm->setKernel( SVM::LINEAR );
    svm->setNu( 0.5 );
    svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
    svm->setC( 0.01 ); // From paper, soft classifier
    //svm->setType( SVM::EPS_SVR ); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task

    return svm;
}

void generate_hog_features(const string filename, vector< float > & descriptor){
    Mat img;
    //cout << "Image filename: " << filename <<endl;
    img = imread(filename);
    if ( !img.data )
    {
        cout << "No image data \n";
        return;
    }
    Mat gray;
    cvtColor( img, gray, COLOR_BGR2GRAY );
    //cout << "Gray image size: " << gray.size() << endl;

    HOGDescriptor hog;
    hog.winSize = gray.size(); 
    hog.blockSize = Size(32, 32);
    hog.cellSize = Size(16, 16);
    hog.blockStride = Size(16, 16);
    hog.nbins = 11;

    // set the descriptor position to the middle of the image
    std::vector<cv::Point> positions; 
    positions.push_back(cv::Point(gray.cols / 2, gray.rows / 2));
    //hog.compute(gray, descriptor, Size(), Size(), positions);
    hog.compute(gray, descriptor, Size(), Size());
}

void get_svm_vector(vector<float> descriptor, input_vector& vec){
    for(int i=0;i<VECTOR_SIZE;i++){
        vec(i) = descriptor[i];
    }
}

void print_vec(vector<float> vec){
    for(float f: vec){
        cout << f << " ";
    }
}

void train_svm(vector<int> pos, vector<int> neg, string cluster_key){
    cout << "Cluster key: " << cluster_key << endl; 
    int total_samples = pos.size() + neg.size();

    vector<input_vector> training_data;
    int count = 0;

    // Adding positive examples to training data
    for(int img_id: pos){
        vector<float> descriptor;
        string filename = "../mpii/cropped_imgs/"+to_string(img_id)+".jpeg";
        generate_hog_features(filename, descriptor);
        input_vector vec;
        //cout << "Pos: "; print_vec(descriptor);
        get_svm_vector(descriptor, vec);
        training_data.push_back(vec);
        count++;
    }

    // Adding negtive examples to training data
    for(int img_id: neg){
        vector<float> descriptor;
        string filename = "../mpii/cropped_imgs/"+to_string(img_id)+".jpeg";
        generate_hog_features(filename, descriptor);
        //cout << "Neg: "; print_vec(descriptor);
        input_vector vec;
        get_svm_vector(descriptor, vec);
        training_data.push_back(vec);
        count++;
    }

    vector<int> labels;
    labels.assign( pos.size(), +1);
    labels.insert( labels.end(), neg.size(), -1);
    cout << "Total number of labels: " << labels.size() << endl;

    dlib::svm_c_linear_trainer<kernel_type> linear_trainer;
    linear_trainer.set_c(10);

    dlib::decision_function<kernel_type> df = linear_trainer.train(training_data, labels);
    dlib::serialize("trained_svms/svm"+cluster_key+".dat") << df;

}


int main(int argc, char** argv){
	// read json file
    ifstream infile { "../jsons/training_data.json" };
    string file_contents { istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };
    auto training_data_map = json::parse(file_contents);
    for(json::iterator it = training_data_map.begin();it != training_data_map.end();it++){
        vector<int> pos_lst = training_data_map[it.key()]["pos"];
        vector<int> neg_lst = training_data_map[it.key()]["neg"];
        train_svm(pos_lst, neg_lst, it.key());
    }
    return 0;
}