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

#include <map>
#include <vector>
#include <cstdlib>

using namespace cv;
using namespace cv::ml;
using namespace std;
using json = nlohmann::json;

#define VECTOR_SIZE 1100

typedef dlib::matrix<double, VECTOR_SIZE, 1> input_vector;
typedef dlib::linear_kernel<input_vector> kernel_type;


struct BoundingBox{
    int x_start;
    int x_end;
    int y_start;
    int y_end;

    BoundingBox(int x_start, int x_end, int y_start, int y_end){
        this->x_start = x_start;
        this->x_end = x_end;
        this->y_start = y_start;
        this->y_end = y_end;
    }

    BoundingBox(){}
};

struct Windows{
    vector<pair<int, int>> x_lst;
    vector<pair<int, int>> y_lst;
};

struct PredictedCluster{
    int key;
    double score;
};

Windows generate_windows(int x_dim, int x_increment, int y_dim, int y_increment, int x_start, int y_start, int x_lim, int y_lim){
    Windows windows;
    int curr_x_start = x_start;
    int curr_x_end = x_start + x_dim;

    while(curr_x_end <= x_lim){
        windows.x_lst.push_back({curr_x_start, curr_x_end});
        curr_x_start += x_increment;
        curr_x_end += x_increment;
    }

    int curr_y_start = y_start;
    int curr_y_end = y_start + y_dim;

    while(curr_y_end <= y_lim){
        windows.y_lst.push_back({curr_y_start, curr_y_end});
        curr_y_start += y_increment;
        curr_y_end += y_increment;
    }

    return windows;
}

Windows get_all_windows(BoundingBox& bounding_box){
    Windows combined_windows;

    int x_width = (bounding_box.x_end - bounding_box.x_start);
    int x_window_size = (int)(x_width/3.0);
    int x_step = (int)(x_window_size/5.0);

    vector<pair<int, int>> x_vals;
    x_vals.push_back({x_window_size, x_step});

    int y_width = (bounding_box.y_end - bounding_box.y_start);
    int y_window_size = (int)(y_width/3.0);
    int y_step = (int)(y_window_size/5.0);

    vector<pair<int, int>> y_vals;
    y_vals.push_back({y_window_size, y_step});


    for(auto x_val: x_vals){
        for(auto y_val: y_vals){
            Windows curr_windows = generate_windows(x_val.first, x_val.second, y_val.first, y_val.second,
                bounding_box.x_start, bounding_box.y_start, bounding_box.x_end, bounding_box.y_end);
            combined_windows.x_lst.insert(combined_windows.x_lst.begin(), curr_windows.x_lst.begin(), curr_windows.x_lst.end());
            combined_windows.y_lst.insert(combined_windows.y_lst.begin(), curr_windows.y_lst.begin(), curr_windows.y_lst.end());
        }
    }

    return combined_windows;
}

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
    //hog.compute(gray, descriptors, Size(), Size(), positions);
    hog.compute(gray, descriptors, Size(), Size());
}

void generate_bounding_box(vector<int> r_hip, vector<int> l_hip, vector<int> head_top, BoundingBox& bounding_box, int x_scale=1, int y_scale=1){
    
    float max_hip_y = max(r_hip[1], l_hip[1]);
    float mean_hip_x = (r_hip[0] + l_hip[0])/2.0;
    float diff_hip_x = abs(r_hip[0] - l_hip[0]);
    float buffer = 0.1 * diff_hip_x;
    int x_start = (int) (mean_hip_x - 0.5 * diff_hip_x * x_scale);
    int x_end = (int) (mean_hip_x + 0.5 * diff_hip_x * x_scale);

    float y_diff = abs(max_hip_y-head_top[1]);
    int y_start = (int)head_top[1];
    int y_end = (int)max_hip_y;
    if(y_end < y_start){
        y_end = max_hip_y; // min_hip_y ?
        y_start = head_top[1];
    }
        
    float y_mean = (y_start+y_end)/2.0;
    y_start = (int) (y_mean - 0.5*y_diff*y_scale);
    y_end = (int) (y_mean + 0.5*y_diff*y_scale);
    
    // x_start, x_end, y_start, y_end
    bounding_box.x_start = x_start;
    bounding_box.x_end = x_end;
    bounding_box.y_start = y_start;
    bounding_box.y_end = y_end;
}

void get_cluster_center_keys(vector<string> &cluster_keys){
    ifstream infile { "../jsons/training_data.json" };
    string file_contents { istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };
    auto training_data_map = json::parse(file_contents);
    for(json::iterator it = training_data_map.begin();it != training_data_map.end();it++){
        cluster_keys.push_back(it.key());
    }
}

void get_svm_vector(vector<float> descriptor, input_vector& vec){
    for(int i=0;i<VECTOR_SIZE;i++){
        vec(i) = descriptor[i];
    }
}

PredictedCluster run_classifiers(vector<float> descriptors){
    //Iterate over classifiers
    float max_score = 0;
    string max_cluster_key = "";
    vector<string> cluster_keys;
    get_cluster_center_keys(cluster_keys);
    for(string cluster_key: cluster_keys){

        dlib::decision_function<kernel_type> df;
        dlib::deserialize("trained_svms/svm"+cluster_key+".dat") >> df;

        
        
        input_vector vec;
        get_svm_vector(descriptors, vec);
        Mat results;
        float df_value = df(vec);
        cout << "Cluster key: " << cluster_key << " ,df_value: " << df_value << endl;  
        if(df_value > max_score){
            max_score = df_value;
            max_cluster_key = cluster_key;
        }
    }

    
    PredictedCluster predicted_cluster;
    predicted_cluster.key = atoi(max_cluster_key.c_str());
    predicted_cluster.score = max_score;
    return predicted_cluster;
}


map<string, pair<int, int>> calculate_distance_with_cluster_pts(int cluster_key, BoundingBox& bounding_box){

    // Read json files
    ifstream infile { "../jsons/new_joint_lst.json" };
    string file_contents { istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };
    vector<map<string, pair<int, int>>> new_joint_lst = json::parse(file_contents);
    map<string, pair<int, int>> predicted_joint_map = new_joint_lst[cluster_key];

    int width = (bounding_box.x_end - bounding_box.x_start);
    int height = (bounding_box.y_end - bounding_box.y_start);
    map<string, pair<int, int>> remapped_predicted_joints;

    string joints[3]{"r_shoulder", "r_elbow", "r_wrist"};

    for(string joint: joints){
        auto predicted_joint = predicted_joint_map[joint]; 
        int remapped_x = (int) (bounding_box.x_start + (predicted_joint.first/96.0)*width);
        int remapped_y = (int) (bounding_box.y_start + (predicted_joint.second/96.0)*height);
        cout << "Key: " << joint << ", x: " << predicted_joint.first << ", y: " << predicted_joint.second << 
            ", remapped_x: " << remapped_x << ", remapped_y: " << remapped_y << endl;
        remapped_predicted_joints[joint] = {remapped_x, remapped_y};
    }

    return remapped_predicted_joints;
}

void display_test_img_with_predicted_joints(string img_name, map<string, pair<int, int>> predicted_joints){
    Mat img = imread("../mpii/images/"+img_name);
    for(auto const& joint: predicted_joints){
        cout << "Key: " << joint.first << ", x: " << joint.second.first << ", y: " << joint.second.second << endl;
        circle(img, Point(joint.second.first, joint.second.second), 2, Scalar(0, 0, 255), 5); 
    }
    imshow("Image with predicted joints", img);
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
    int test_id = 0;
    Mat img = imread("../mpii/images/"+img_lst[test_id]);
    imshow("Test Image", img);
    waitKey(0);

    // Extract bounding boxes co-ordinate from image
    BoundingBox bounding_box;
    generate_bounding_box(joint_lst[test_id]["r_hip"], joint_lst[test_id]["l_hip"], joint_lst[test_id]["head_top"], bounding_box, 3.0, 3.0);
    cout << "Bounding boxes: " << bounding_box.x_start << " " << bounding_box.x_end << 
        " " << bounding_box.y_start << " " << bounding_box.y_end << endl;   

    int predicted_pose_cluster = 0;
    float max_score = 0;
    BoundingBox max_score_window;

    Windows windows = get_all_windows(bounding_box);
    for(auto x_range: windows.x_lst){
        for(auto y_range: windows.y_lst){
            // Cut image sub section with bounding box
            Rect roi(x_range.first, y_range.first, 
                (x_range.second - x_range.first), (y_range.second - y_range.first));
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
            PredictedCluster predicted_cluster = run_classifiers(descriptors);

            if(max_score < predicted_cluster.score){
                predicted_pose_cluster = predicted_cluster.key;
                max_score = predicted_cluster.score;
                max_score_window = BoundingBox(x_range.first, x_range.second,
                    y_range.first, y_range.second);
            }
        }
    }
    cout << "Max cluster key: "  << predicted_pose_cluster << endl;

    // Print image with max id
    Mat max_cluster_center_img = imread("../mpii/cropped_imgs/" + to_string(predicted_pose_cluster) + ".jpeg");
    imshow("Best cluster center image", max_cluster_center_img);
    waitKey(0);

    // Show cluster center image of best match classifier 
    map<string, pair<int, int>> predicted_joints = calculate_distance_with_cluster_pts(predicted_pose_cluster, max_score_window);
    display_test_img_with_predicted_joints(img_lst[test_id], predicted_joints);


    return 0;
}

