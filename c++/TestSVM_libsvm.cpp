#include "json.hpp"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

#include <map>
#include <vector>
//#include <experimental/filesystem>
//#include <filesystem>


using namespace cv;
using namespace cv::ml;
using namespace std;
using json = nlohmann::json;

struct BoundingBox{
	int x_start;
	int x_end;
	int y_start;
	int y_end;
};

void generate_hog_features(Mat& img, vector< float > & descriptors){
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

void generate_bounding_box(vector<int> r_hip, vector<int> l_hip, vector<int> head_top, BoundingBox& bounding_box){
	
	float max_hip_y = max(r_hip[1], l_hip[1]);
    float mean_hip_x = (r_hip[0] + l_hip[0])/2.0;
    float diff_hip_x = abs(r_hip[0] - l_hip[0]);
    float buffer = 0.1 * diff_hip_x;
    int x_start = (int) (mean_hip_x - 0.5 * diff_hip_x - buffer);
    int x_end = (int) (mean_hip_x + 0.5*diff_hip_x + buffer);
    
   	// x_start, x_end, y_start, y_end
    bounding_box.x_start = x_start;
    bounding_box.x_end = x_end;
    bounding_box.y_start = head_top[1];
    bounding_box.y_end = (int) max_hip_y;
}

void get_cluster_center_keys(vector<string> &cluster_keys){
	ifstream infile { "../jsons/training_data.json" };
    string file_contents { istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };
    auto training_data_map = json::parse(file_contents);
    for(json::iterator it = training_data_map.begin();it != training_data_map.end();it++){
    	cluster_keys.push_back(it.key());
    }
}

void run_classifiers(vector<float> descriptors){
	//Iterate over classifiers
	float max_score = 0;
	string max_cluster_key = "";
	vector<string> cluster_keys;
	get_cluster_center_keys(cluster_keys);
	for(string cluster_key: cluster_keys){
		Ptr<SVM> svm = Algorithm::load<SVM>("trained_svms/svm_"+cluster_key+".xml");
		Mat predict1D(1, descriptors.size(), CV_32F, &descriptors[0]);
		int res = svm->predict(predict1D);
		cout << "Predicted class: " << res << endl;
		//float df_value = svm->predict(predict1D, true);
		/*Mat results;
		float df_value = svm->predict(predict1D, results, StatModel::RAW_OUTPUT);
		cout << "Results: " << results.size() << endl; 
		cout << results.at<float>(0, 0) << endl;
		cout << "Cluster key: " << cluster_key << " ,df_value: " << df_value << endl;  
		if(df_value > max_score){
			max_score = df_value;
			max_cluster_key = cluster_key;
		}*/
	}

	cout << "Max cluster key: "  << max_cluster_key << endl;

	// Print image with max id
	Mat max_cluster_center_img = imread("../mp2/cropped_imgs/" + max_cluster_key + ".jpeg");
	imshow("Display Image", max_cluster_center_img);
    waitKey(0);
}

int main(int argc, char** argv ){
	cout << "I am in test SVM" << endl;
	
	// Read json files
    ifstream infile { "../jsons/img_data.json" };
    string file_contents { istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };
    auto img_data_map = json::parse(file_contents);
    vector<string> img_lst = img_data_map["img_lst"];
    vector<map<string, vector<int>>> joint_lst = img_data_map["orig_joint_lst"];

    // Read one image from list
    Mat img = imread("../mp2/images/"+img_lst[0]);

	// Extract bounding boxes co-ordinate from image
	BoundingBox bounding_box;
	generate_bounding_box(joint_lst[0]["r_hip"], joint_lst[0]["l_hip"], joint_lst[0]["head_top"], bounding_box);
	cout << "Bounding boxes: " << bounding_box.x_start << " " << bounding_box.x_end << 
		" " << bounding_box.y_start << " " << bounding_box.y_end << endl;   

	// Cut image sub section with bounding box
	Rect roi(bounding_box.x_start, bounding_box.y_start, 
		(bounding_box.x_end - bounding_box.x_start), (bounding_box.y_end - bounding_box.y_start));
	Mat cropped_image = img(roi);
	//roi.copyTo(cropped_image);

	// Resize image
	Mat resized_img;
	resize(cropped_image, resized_img, Size(96, 96), 0, 0, CV_INTER_LINEAR);

	// Generate HOG features for bounding box
	vector<float> descriptors;
	generate_hog_features(resized_img, descriptors);
	cout << "Size of descriptor vector: " << descriptors.size() << endl;

	// Run SVM classifier on the bounding box 
	run_classifiers(descriptors);
	// Show cluster center image of best match classifier 
	return 0;
}

