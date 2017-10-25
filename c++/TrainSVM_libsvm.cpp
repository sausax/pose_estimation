#include "json.hpp"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>


using namespace cv;
using namespace cv::ml;
using namespace std;
using json = nlohmann::json;


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

void generate_hog_features(const string filename, vector< float > & descriptors){
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
    std::vector<float> descriptor;
    hog.compute(gray, descriptors, Size(), Size(), positions);
}

void train_svm(vector<int> pos, vector<int> neg, string cluster_key){
//void train_svm(Mat train_data){

    cout << "Cluster key: " << cluster_key << endl; 
    int total_samples = pos.size() + neg.size();

    Mat training_data = Mat( total_samples, 1100, CV_32FC1 );
    int count = 0;

    // Adding positive examples to training data
    for(int img_id: pos){
        vector<float> descriptor;
        string filename = "../mp2/cropped_imgs/"+to_string(img_id)+".jpeg";
        generate_hog_features(filename, descriptor);
        Mat descriptor_mat = Mat(descriptor);
        Mat tmp;
        transpose(descriptor_mat, tmp);
        tmp.copyTo(training_data.row(count));
        count++;
    }

    // Adding negtive examples to training data
    for(int img_id: neg){
        vector<float> descriptor;
        string filename = "../mp2/cropped_imgs/"+to_string(img_id)+".jpeg";
        generate_hog_features(filename, descriptor);
        Mat descriptor_mat = Mat(descriptor);
        Mat tmp;
        transpose(descriptor_mat, tmp);
        tmp.copyTo(training_data.row(count));
        count++;
    }

  
    vector<int> labels;
    labels.assign( pos.size(), +1 );
    labels.insert( labels.end(), neg.size(), -1 );

    Ptr<SVM> svm = initialize_svm();
    svm->train( training_data, ROW_SAMPLE, Mat(labels));
    svm->save("trained_svms/svm_"+cluster_key+".xml");

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
	// get the ids of pos, neg example
	// extract hog features
	// train svm classifier	
    /*vector<float> descp;
    generate_hog_features("../mp2/cropped_imgs/0.jpeg", descp);
    cout << "Feature vector size: " << descp.size() << endl;
    Mat train_data = Mat( 4, 1100, CV_32FC1 );
;
    Mat descp_mat = Mat(descp);
    cout << "Copying elements " << endl;
    cout << "Size of descp_mat: " << descp_mat.size() << endl;
    Mat tmp;
    transpose(descp_mat, tmp);
    cout << "Size of transpose matrix: " << tmp.size() << endl;
    tmp.copyTo(train_data.row(0));
    tmp.copyTo(train_data.row(1));
    tmp.copyTo(train_data.row(2));
    tmp.copyTo(train_data.row(3));

    train_svm(train_data);
    */
    return 0;
}