/*System includes*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#include <signal.h>
#include <iostream>
/*Application includes*/
//#include "face_rec_server.h"
//#include "face_rec_utils.h"
#include "opencv2/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui.hpp"
//#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

/********************************************************************************************/
/*	Macros										    */
/*											    */
/*											    */
/********************************************************************************************/


#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

// Function Headers
void detectAndDisplay(Mat frame);

// Global variables
// Copy this file from opencv/data/haarscascades to target folder
string face_cascade_name = "haarcascade_frontalface_default.xml";
CascadeClassifier face_cascade;


// Function main
int detect_face(void)
{
    // Load the cascade
    if (!face_cascade.load(face_cascade_name)){
        printf("--(!)Error loading\n");
        return (-1);
    }
    
    // Read the image file
    Mat frame = imread("km_closeid.jpg");
    
    // Apply the classifier to the frame
    if (!frame.empty()){
        detectAndDisplay(frame);
    }
    
    return 0;
}

// Function detectAndDisplay
void detectAndDisplay(Mat frame)
{
    std::vector<Rect> faces;
    Mat frame_gray;
    Mat crop;
    Mat res;
    Mat gray;
    string text;
    stringstream sstm;
    
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    
    // Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(500, 500));
    
    // Set Region of Interest
    cv::Rect roi_b;
    cv::Rect roi_c;
    
    size_t ic = 0; // ic is index of current element
    int ac = 0; // ac is area of current element
    
    size_t ib = 0; // ib is index of biggest element
    int ab = 0; // ab is area of biggest element
    int filenumber = 0;
    for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)
        
    {
        roi_c.x = faces[ic].x;
        roi_c.y = faces[ic].y;
        roi_c.width = (faces[ic].width);
        roi_c.height = (faces[ic].height);
        
        ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)
        
        roi_b.x = faces[ib].x;
        roi_b.y = faces[ib].y;
        roi_b.width = (faces[ib].width);
        roi_b.height = (faces[ib].height);
        
        ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element
        
        if (ac > ab)
        {
            ib = ic;
            roi_b.x = faces[ib].x;
            roi_b.y = faces[ib].y;
            roi_b.width = (faces[ib].width);
            roi_b.height = (faces[ib].height);
        }
        
        crop = frame(roi_b);
        resize(crop, res, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
        cvtColor(crop, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale
        
        // Form a filename
        string filename = "";
        stringstream ssfn;
        ssfn << filenumber << ".jpg";
        filename = ssfn.str();
        filenumber++;
        
        imwrite(filename, gray);
        
        //Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window - live stream from camera
        //Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
        //rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
    }
    
}




/********************************************************************************************/
/*	Function declarations								    */
/*											    */
/*											    */
/********************************************************************************************/



extern char serv_err_strings[][512];


    struct timeval timeout;

#define NUM_IMG_ROWS (243)
#define NUM_IMG_COLS (320) 

/********************************************************************************************/
/*	Application Main								    */
/*											    */
/*											    */
/********************************************************************************************/
int main(void)
{
    std::string modelName;
    std::vector<Mat> _images;
    std::vector<int> _labels;

    Ptr<FaceRecognizer> pModel =  createFisherFaceRecognizer();
    //Ptr<FaceRecognizer> pModel  = createEigenFaceRecognizer(80);
    

    std::string filenames[12];
    filenames[0] = "live1.jpg";
    filenames[1] = "live2.jpg";
    filenames[2] = "live3.jpg";
    filenames[3] = "live4.jpg";
    filenames[4] = "livea.jpg";
    filenames[5] = "liveb.jpg";
    filenames[6] = "livec.jpg";
    filenames[7] = "lived.jpg";
    filenames[8] = "km1.jpg";
    filenames[9] = "km2.jpg";
    filenames[10] = "km3.jpg";
    filenames[11] = "km4.jpg";
    
    detect_face();
    //return 0;
    
    int predictedLabel = -1;
    try{
    Mat im_in;
    for ( size_t i = 0 ; i < 4; i++) {
        im_in = imread(filenames[i], CV_LOAD_IMAGE_GRAYSCALE);
        /*Resize each image*/
        Mat im;
        im.rows = NUM_IMG_ROWS; im.cols = NUM_IMG_COLS;
        
        cv::resize(im_in, im, im.size(), 0,0,INTER_LANCZOS4);
        _images.push_back(im);
        _labels.push_back(1);
    }
    
    for ( size_t i = 4 ; i < 8; i++) {
        im_in = imread(filenames[i], CV_LOAD_IMAGE_GRAYSCALE);
        /*Resize each image*/
        Mat im;
        im.rows = NUM_IMG_ROWS; im.cols = NUM_IMG_COLS;
        
        cv::resize(im_in, im, im.size(), 0,0,INTER_LANCZOS4);
        _images.push_back(im);
        _labels.push_back(2);
    }
#if 1
    for ( size_t i = 8 ; i < 12; i++) {
        im_in = imread(filenames[i], CV_LOAD_IMAGE_GRAYSCALE);
        /*Resize each image*/
        Mat im;
        im.rows = NUM_IMG_ROWS; im.cols = NUM_IMG_COLS;
            
        cv::resize(im_in, im, im.size(), 0,0,INTER_LANCZOS4);
        _images.push_back(im);
        _labels.push_back(3);
    }
#endif
    pModel->train(_images,_labels);
    

    Mat im_test;
    std::string testfile = "0.jpg";
    im_test = imread(testfile, CV_LOAD_IMAGE_GRAYSCALE);
    
    Mat im_test_resized;
    im_test_resized.rows = NUM_IMG_ROWS; im_test_resized.cols = NUM_IMG_COLS;
    
    cv::resize(im_test, im_test_resized, im_test_resized.size(), 0,0,INTER_LANCZOS4);
    predictedLabel = pModel->predict(im_test_resized);
    } catch (cv::Exception& e) {
        cout << "Face recognition:Error reading image "  << "\". Reason: " << e.msg << endl;
        std::cerr << "Face recognition:Error reading image " << "\". Reason: " << e.msg << endl;
    }
    
    
    printf("exiting %d\n",predictedLabel);
#if 0
    int sockfd, new_fd;  // listen on sock_fd, new connection on new_fd
    struct addrinfo hints, *servinfo, *p;
    struct sockaddr_storage their_addr; // connector's address information
    socklen_t sin_size;
    struct sigaction sa;
    int yes=1;
    char s[INET6_ADDRSTRLEN];
    int rv;
    int len;
    char cmd_buffer[MAX_CLIENT_COMMAND_LEN];
    timeout.tv_sec = 5;
    timeout.tv_usec = 0;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE; // use my IP


   /*Set STDERR*/
   std::ofstream serverLog;
   serverLog.open ("FaceRecServer.log");

   std::streambuf * streamBuffer = serverLog.rdbuf();
   std::cerr.rdbuf(streamBuffer);


    if ((rv = getaddrinfo(NULL, PORT, &hints, &servinfo)) != 0) {
	 std::cerr << "getaddrinfo:" << gai_strerror(rv) << endl;
        return 1;
    }

    // loop through all the results and bind to the first we can
    for(p = servinfo; p != NULL; p = p->ai_next) {
        if ((sockfd = socket(p->ai_family, p->ai_socktype,
                p->ai_protocol)) == -1) {
            perror("server: socket");
            continue;
        }

        if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &yes,
                sizeof(int)) == -1) {
            perror("setsockopt");
            exit(1);
        }

        if (bind(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
            close(sockfd);
            perror("server: bind");
            continue;
        }

        break;
    }

    if (p == NULL)  {
        std::cerr << "Server failed to bind"  << endl;
        exit(1);
    }

    freeaddrinfo(servinfo); // all done with this structure

    if (listen(sockfd, BACKLOG) == -1) {
        perror("listen");
        exit(1);
    }

    sa.sa_handler = sigchld_handler; // reap all dead processes
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    if (sigaction(SIGCHLD, &sa, NULL) == -1) {
        perror("sigaction");
        exit(1);
    }
	
    cout << "Face recognition: Waiting for commands"  << endl << endl;
    std::cerr << "Face recognition: Waiting for commands"  << endl << endl;

    /*create a face recognition server*/
    FaceRecServer *server = new FaceRecServer(10);
    if(NULL == server){
		cout << "Face recognition: Server creation failed " << endl;
   		std::cerr << "Face recognition: Server creation failed "   << endl;
    }

    while(1) {	
     	/*main accept() loop*/
        sin_size = sizeof their_addr;
        new_fd = accept(sockfd, (struct sockaddr *)&their_addr, &sin_size);
        if (new_fd == -1) {
            perror("accept");
            continue;
        }

        inet_ntop(their_addr.ss_family,
            get_in_addr((struct sockaddr *)&their_addr),
            s, sizeof s);
   
	/*Receive*/  	
	len = recv(new_fd, &cmd_buffer, MAX_CLIENT_COMMAND_LEN, 0);
	cmd_buffer[len-1] = '\0';

   	cout << "Face Recognizer: Received command "  << cmd_buffer << endl;
   	std::cerr << "Face Recognizer: Received command "  << cmd_buffer << endl;




	/***********Parse Command()************************************ */
	Request *request;
	enum_err_server status = server->parseClientRequest(&cmd_buffer[0], &request);

   	cout << "Face Recognizer: reqType = "  << request->_requestType << endl;
   	std::cerr << "Face Recognizer: reqType = "  << request->_requestType << endl;
	

	/*If status fail, send response*/
	if(status != SERV_SUCCESS){
		cout << "Face Recognizer: failed to parse client request "  << serv_err_strings[status] << endl << endl;
   		std::cerr << "Face Recognizer: failed to parse client request "  << serv_err_strings[status] << endl << endl;

        	if (send(new_fd, serv_err_strings[status], strlen(serv_err_strings[status]), 0) == -1){
            	 	perror("send");
		}
		continue;
	}


	/***********Execute Command()************************************ */
	string replyMessage = "Processed Request";
	if(request != NULL){
		status = server->ProcessRequest(request, replyMessage);
		cout << "Face Recognizer: Processed request, status =  "  << serv_err_strings[status] << "," << replyMessage << endl << endl;
   		std::cerr << "Face Recognizer: Processed request, status =  "  << serv_err_strings[status] << "," << replyMessage << endl << endl;
	}



	/***********Send Response()************************************ */
	if(status == SERV_SUCCESS){
		if (send(new_fd, replyMessage.c_str(),strlen(replyMessage.c_str()), 0) == -1){
            	 	perror("send");
		}
	}
	else{
		if (send(new_fd, serv_err_strings[status], strlen(serv_err_strings[status]), 0) == -1){
            	 	perror("send");
		}
	}

/****************************************************************************Concurrent server*/
#if 0 
        if (!fork()) { // this is the child process
            close(sockfd); // child doesn't need the listener
            if (send(new_fd, "Hello, world!", 13, 0) == -1)
                perror("send");
            close(new_fd);
            exit(0);
        }
#endif
        //close(new_fd);  // parent doesn't need this
/*****************************************************************************/


    }/*while(1)*/

#endif

    return 0;
}


