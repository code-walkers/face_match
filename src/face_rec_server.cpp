
/**************************face_rec_server.cpp*****************************/

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

/*Application includes*/ 
#include "face_rec_server.h"
#include "face_rec_utils.h"

/********************************************************************************************/
/*	Macros										    */
/*											    */
/*											    */
/********************************************************************************************/



/********************************************************************************************/
/*	Function declarations								    */
/*											    */
/*											    */
/********************************************************************************************/


/********************************************************************************************/
/*	Constants									    */
/*											    */
/*											    */
/********************************************************************************************/

FaceRecModel INVALID_FACEREC_MODEL("INVALID");


/********************************************************************************************/
/*	Error strings returned by server						    */
/*											    */
/*											    */
/********************************************************************************************/
char serv_err_strings[SERV_ERR_MAX][512] =
{
	"Executed request successfully",
	"Unrecognized Command",
	"CSV file not found",
	"Model Name not specified",
	"Input Image not specified",
	"Error processing CSV file",
	"Error creating model",
	"Not enough images",
	"Error creating Eigen face recognizer",
	"Failed to open file",
	"Model not found",
	"Failed to open Image",
	"Error loading Eigen Model",
	"Training to create Eigen Model failed",
	"Saving of model failed",
	"Resizing image failed",
	"Unsupported file format",
	"Server internal failure. Failed to open tmp image to convert base64",

};




/********************************************************************************************/
/*	enum_err_server FaceRecServer::recognizeImage					    */
/*											    */
/*											    */
/********************************************************************************************/
enum_err_server FaceRecServer::recognizeImage(string inputImage, string modelName, int &label)
{
	string recognition_jpg ;
	enum_err_server status;
	FaceRecModel *pmdl;
	bool is_input_base64 = true;	
	string outb64DecdJpeg("temp64decd.jpg");

	/*Load model if not already loaded. If already loaded, do not reload it*/
	status = LoadModel(modelName, false);

	/*Confirm that model was loaded successfully*/
	FaceRecModel mdl = searchModelsByModelName(modelName);

	if(mdl == INVALID_FACEREC_MODEL){	
		cout << "Recognize: Failed to load model  "  << modelName << endl;
   		std::cerr << "Recognize: Failed to load model  "  << modelName << endl;
		return status;
	}

	
	/*If input is base 64, Convert base 64 images to jpg*/
  	string b64InputImageString; /* base 64 string*/

  	ifstream infile_b64(inputImage.c_str(), std::ios::in | std::ios::binary);

	if(!infile_b64.is_open()){
		cout << "Face recognition: Failed to open temp base64 image "  << endl;
   		std::cerr << "Face recognition: Failed to open temp base64 image "  << endl;
		return SER_ERR_FAIL_OPEN_TMP_FILE;
	}

  	ofstream outfile_jpg(outb64DecdJpeg.c_str(), std::ios::out | std::ios::binary);

	if(!outfile_jpg.is_open()){
		cout << "Face recognition: Failed to open temp jpg image for base64 decode "  << endl;
   		std::cerr << "Face recognition: Failed to open temp jpg image for base64 decode "  << endl;
		/*clean up*/		
		infile_b64.close();
		return SER_ERR_FAIL_OPEN_TMP_FILE;
	}

	/*Read the input file into a string*/
  	if (infile_b64)
  	{
   		infile_b64.seekg(0, std::ios::end);
   		b64InputImageString.resize(infile_b64.tellg());
   		infile_b64.seekg(0, std::ios::beg);
    		infile_b64.read(&b64InputImageString[0],  b64InputImageString.size());
		/*clean up*/    		
		infile_b64.close();
 	}

	/*Decode base 64 string*/  		
	std::string decoded_img_string = base64_decode( b64InputImageString);

	/*Check if the image was base 64*/
 	if(decoded_img_string.empty()){
		is_input_base64 = false;
		cout << "Face recognition: Not a base 64 image "  << inputImage << endl;
   		std::cerr << "Face recognition: Not a base 64 image "  << inputImage << endl;
  	}

	/*Close the file after writing the decoded image to it*/
  	outfile_jpg << decoded_img_string;
  	outfile_jpg.close();


	/*end of convert base64 to jpg*/
	if(is_input_base64 == true){
		cout << "Face recognition: Is a base 64 image "  << inputImage << endl;
   		std::cerr << "Face recognition: Is a base 64 image "  << inputImage << endl;
		recognition_jpg = outb64DecdJpeg;
	}
	else{
		recognition_jpg = inputImage;
	}


	/*Read the recognition image in*/
	Mat im_in;
	try {		
		im_in = imread(recognition_jpg.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    
	} catch (cv::Exception& e) {
		
		cout << "Face recognition:Error reading image " << recognition_jpg << "\". Reason: " << e.msg << endl;
		std::cerr << "Face recognition:Error reading image " << recognition_jpg << "\". Reason: " << e.msg << endl;

		return SERV_ERR_FAIL_OPEN_IMG;
	}
	

	if(im_in.data == NULL){
		cout << "Face recognition:Unsupported file format detected " << recognition_jpg <<  endl;
		std::cerr << "Face recognition:Unsupported file format detected " << recognition_jpg <<  endl;
		return SERV_ERR_UNSUPP_FILE_FMT;
	}
	
	Mat im;
	im.rows = NUM_IMG_ROWS; im.cols = NUM_IMG_COLS;


	/*Resize image*/

	try {
 		cv::resize(im_in, im, im.size(), 0,0,INTER_LANCZOS4);
	}catch (cv::Exception& e) {
		cout << "Face recognition:Failed to resize image" << recognition_jpg << "\". Reason: " << e.msg << endl;
		std::cerr << "Face recognition:Failed to resize image" << recognition_jpg << "\". Reason: " << e.msg << endl;
		return SERV_ERR_RESIZE_IMG_FAIL;
	}

	/*Predict the label*/
  	int predictedLabel = -1;
	try {
		predictedLabel = mdl.model_core->predict(im);
	}catch (cv::Exception& e) {
		cout << "Face recognition:Prediction exception model" << mdl._modelName << " image " << recognition_jpg << "\". Reason: " << e.msg << endl;
		std::cerr << "Face recognition:Prediction exception"  << mdl._modelName << " image " << recognition_jpg << "\". Reason: " << e.msg << endl;
	}

	label = predictedLabel;

 
	cout << "Face recognition:Predicted class " << predictedLabel << endl;
	std::cerr << "Face recognition:Predicted class" << predictedLabel << endl;

	return SERV_SUCCESS;

}
/********************************************************************************************/
/*	enum_err_server FaceRecServer::TrainModel					    */
/*											    */
/*											    */
/********************************************************************************************/

enum_err_server FaceRecServer:: TrainModel(string filenamesCsv, string modelName, enum_face_detect_method algorithm)
{ 
	string outputModelFile(modelName);

	FaceRecModel *pModel = new FaceRecModel(modelName);
		 	
	/*If model creation failed*/
	if(pModel == NULL){
		
		cout << "TrainModen: Failed to create model  "  << modelName << endl;
   		std::cerr << "TrainModen: Failed to create model  "  << modelName << endl;

		return SERV_ERR_CREAT_MODEL;
	}
    	
	/*Read in the data. This can fail if no valid, input filename is given.*/
    	try {       
		read_csv(filenamesCsv, pModel->_images, pModel->_labels);
    
	} catch (cv::Exception& e) {
       		
		/*cleanup*/
		cerr << "Error processing file \"" << filenamesCsv << "\". Reason: " << e.msg << endl;

        	if(pModel != NULL){
			
			delete pModel;

		}
        	return SERV_ERR_PROCESSING_CSV;
    	}

    	/* Quit if there are not enough images.*/
    	if(pModel->_images.size() <= 1) {
        	
		if(pModel != NULL){			
			delete pModel;
		}
		return SERV_ERR_NO_IMAGES;		
    	}

	/*Create Eigen face recognizer*/
	if(algorithm == FD_EIGEN_FACES){
    		try {       
			pModel->model_core = createEigenFaceRecognizer(80);
    
		} catch (cv::Exception& e) {
		
			/*cleanup*/
			cerr << "Error creating Eignen Face Recognizer \"" << pModel->_modelName << "\". Reason: " << e.msg << endl;
			if(pModel != NULL){			
				delete pModel;
			}
			return SERV_ERR_CREATION_EIGEN_FAIL;
		}
		/*If empty model, return*/
		if( pModel->model_core.empty() ){		
			
			if(pModel != NULL){			
				delete pModel;
			}
			return SERV_ERR_CREATION_EIGEN_FAIL;												
		}		
	}

	
	/*Train the model*/
	try {		
		pModel->model_core->train(pModel->_images,pModel->_labels);
    
	} catch (cv::Exception& e) {
		
		/*cleanup*/
		cerr << "Error training model \"" << pModel->_modelName << "\". Reason: " << e.msg << endl;
		if(pModel != NULL){			
			delete pModel;
		}
		return SERV_ERR_FAIL_TRAIN_EIGEN;
	}		


	/*Open file storgage to store model*/
	FileStorage fs;

	try {		
		fs.open(outputModelFile, FileStorage::WRITE);
    
	} catch (cv::Exception& e) {
		
		/*cleanup*/
		cerr << "Error opening file to store \"" << pModel->_modelName << "\". Reason: " << e.msg << endl;
		if(pModel != NULL){			
			delete pModel;
		}
		return SERV_ERR_FAIL_OPEN_STORAGE;
	}

	if(!fs.isOpened()){
		if(pModel != NULL){			
			delete pModel;
		}
		return SERV_ERR_FAIL_OPEN_STORAGE;
	}


	/*Save the model*/
	try {
		pModel->model_core->save(fs);

	}catch (cv::Exception& e) {
		
		/*cleanup*/
		cerr << "Error saving model \"" << pModel->_modelName << "\". Reason: " << e.msg << endl;
		if(pModel != NULL){			
			delete pModel;
		}
		fs.release();
		return SERV_ERR_FAIL_SAVE_MODEL;
	}

	/*release file storage*/
	fs.release();

	/*Add model to list of models on server*/
	//models.push_back(pModel);

	if(pModel != NULL){
		delete(pModel);
	}

	return SERV_SUCCESS;

}


/********************************************************************************************/
/*	enum_err_server FaceRecServer::searchModelsByModelName				    */
/*											    */
/*											    */
/********************************************************************************************/

FaceRecModel & FaceRecServer::searchModelsByModelName(string modelName)
{
	for (std::vector<FaceRecModel *>::iterator it = models.begin() ; it != models.end(); ++it)
	{
		if((*it)->_modelName == modelName){

			cout << "searchModelsByModelName: Found model =  "  << modelName << endl;
   			std::cerr << "searchModelsByModelName: Found model =  "  << modelName << endl;
			
			return (**it);
		}
	}

	cout << "searchModelsByModelName: Model Not found =  "  << modelName << endl;
   	std::cerr << "searchModelsByModelName:  Model Not found =  "  << modelName << endl;

	return INVALID_FACEREC_MODEL;
}

/********************************************************************************************/
/*	enum_err_server FaceRecServer::ListLoadedModels					    */
/*											    */
/*											    */
/********************************************************************************************/

enum_err_server FaceRecServer::ListLoadedModels(string &loaded_models)
{
	stringstream replyStringStream;

	for (std::vector<FaceRecModel *>::iterator it = models.begin() ; it != models.end(); ++it) {
		replyStringStream << (*it)->_modelName << endl;
	}
	
	loaded_models = replyStringStream.str();
	
	return SERV_SUCCESS;
}

/********************************************************************************************/
/*	enum_err_server FaceRecServer::UnLoadModel					    */
/*											    */
/*											    */
/********************************************************************************************/

enum_err_server FaceRecServer::UnloadModel(string modelName)
{
	for (std::vector<FaceRecModel *>::iterator it = models.begin() ; it != models.end(); ++it)
	{
		if((*it)->_modelName == modelName){

			cout << "UnloadModel: Unloading model "  << modelName << endl;
   			std::cerr << "UnloadModel: Unloading model "  << modelName << endl;
			delete(*it);
			models.erase(it);

			return SERV_SUCCESS;
		}
	}

	cout << "UnloadModel: Model not found "  << modelName << endl;
   	std::cerr << "UnloadModel: Model not found "  << modelName << endl;

	return SERV_ERR_MODEL_NOT_FOUND;
}

/********************************************************************************************/
/*	enum_err_server FaceRecServer::LoadModel					    */
/*											    */
/*											    */
/********************************************************************************************/

enum_err_server FaceRecServer:: LoadModel(string modelName, bool reload)
{
	enum_err_server status;

	cout << "LoadModel: Searching models for "  << modelName << endl;
   	std::cerr << "LoadModel: Searching models for "  << modelName << endl;

	if(reload == true){

		status = UnloadModel(modelName);

		if(status == SERV_SUCCESS){
			cout << "LoadModel: Unloaded "  << modelName << endl;
   			std::cerr << "LoadModel: Unloaded "  << modelName << endl;
		}		
	}

	FaceRecModel mdl = searchModelsByModelName(modelName);

	if( /* FaceRecModel::*/INVALID_FACEREC_MODEL == mdl){
		cout << "LoadModel: model not yet loaded =  " << modelName << endl;
   		std::cerr << "LoadModel: model not yet loaded =  " << modelName << endl;
		
		FaceRecModel *pModel = new FaceRecModel(modelName);
		
		if(pModel == NULL){
			cout << "LoadModel: failed to create model  " << modelName << endl;
   			std::cerr << "LoadModel: failed to create model  " << modelName << endl;
			return SERV_ERR_CREAT_MODEL;
		}
			
		/*Open the model*/
		FileStorage fs;
		try {
			fs.open(modelName,FileStorage::READ);
		}catch (cv::Exception& e) {		
			/*cleanup*/
			cerr << "LoadModel:Failed to open file  \"" << modelName << "\". Reason: " << e.msg << endl;
			cout << "LoadModel:Failed to open file  \"" << modelName << "\". Reason: " << e.msg << endl;

			if(pModel != NULL){			
				delete pModel;
			}
			return SERV_ERR_FAIL_OPEN_STORAGE;	
		}
		
		if(!fs.isOpened()){
			if(pModel != NULL){			
				delete pModel;
			}
			return SERV_ERR_FAIL_OPEN_STORAGE;
		}
		
		/*Create Eigen face recognizer*/
		try {			
			pModel->model_core = createEigenFaceRecognizer();
		}catch (cv::Exception& e) {		
			/*cleanup*/
			cerr << "Error creating eigen model \"" << pModel->_modelName << "\". Reason: " << e.msg << endl;
			cout << "Error creating eigen model \"" << pModel->_modelName << "\". Reason: " << e.msg << endl;
			if(pModel != NULL){			
				delete pModel;
			}
			return SERV_ERR_CREATION_EIGEN_FAIL;	
		}
		
		/*Cleanup and return if face recognizer is not created*/
		if( pModel->model_core.empty() ){			
			cout << "LoadModel: failed to create recognizer  " << modelName << endl;
   			std::cerr << "LoadModel: failed to create recognizer  " << modelName << endl;
			/*Cleanup*/
			if(NULL != pModel){
				delete pModel;
			}

			fs.release();
			return SERV_ERR_CREATION_EIGEN_FAIL;				
								
		}

		/*Load model*/
		try {			
			pModel->model_core->load(fs);

		}catch (cv::Exception& e) {		
			/*cleanup*/
			cerr << "Error loading model \"" << pModel->_modelName << "\". Reason: " << e.msg << endl;
			cout << "Error loading model \"" << pModel->_modelName << "\". Reason: " << e.msg << endl;
			if(pModel != NULL){			
				delete pModel;
			}
			fs.release();
			return SERV_ERR_LOADING_EIG_MODEL;
		}
		

		models.push_back(pModel);
		
		cout << "LoadModel: Loaded Model  "  << modelName << endl;
   		std::cerr << "LoadModel: Loaded Model  "  << modelName << endl;
		
		/*Close the file storage. Model is loaded now*/
		fs.release();	
		
	}
	else{
		cout << "LoadModel: Model already loaded "  << modelName << endl;
   		std::cerr  << "LoadModel: Model already loaded "  << modelName << endl;
	}

	return SERV_SUCCESS;
}


/********************************************************************************************/
/*	enum_err_server FaceRecServer::ProcessRequest					    */
/*											    */
/*											    */
/********************************************************************************************/

enum_err_server FaceRecServer::ProcessRequest(Request *request, string &replyMessage)
{
	enum_err_server status = SERV_SUCCESS;

	switch(request->_requestType)
	{
		case SERV_REQ_TRAIN:
		{
			
			TrainRequest *trainReq = static_cast<TrainRequest*>(request);
			status = TrainModel(trainReq->_inputCSVFileName, trainReq->_outputModelName);

			if(status == SERV_SUCCESS){
				stringstream replyStringStream;
				replyStringStream << "Trained & stored (not loaded) model  " << trainReq->_outputModelName << " using " << trainReq->_inputCSVFileName ;
				replyMessage = replyStringStream.str();
			}

			delete(request);
			break;
		}
		case SERV_REQ_LOAD:
		{
			
			LoadModelRequest *loadReq = static_cast<LoadModelRequest*>(request);
			status = LoadModel(loadReq->_modelName, true);

			if(status == SERV_SUCCESS){
				stringstream replyStringStream;
				replyStringStream << "Loaded " << loadReq->_modelName;
				replyMessage = replyStringStream.str();
			}

			delete(request);
			break;
		}
		case SERV_REQ_UNLOAD:
		{
			
			UnloadModelRequest *unloadReq = static_cast<UnloadModelRequest*>(request);
			status = UnloadModel(unloadReq->_modelName);

			if(status == SERV_SUCCESS){
				stringstream replyStringStream;
				replyStringStream << "Unloaded " << unloadReq->_modelName;
				replyMessage = replyStringStream.str();
			}

			delete(request);
			break;
		}
		case SERV_REQ_RECOGNIZE:
		{
			int label = -1;
			RecognizeRequest *recogRequest = static_cast<RecognizeRequest*>(request);
			
			status = recognizeImage(recogRequest->_inputImage,recogRequest->_modelName, label);

			if(status == SERV_SUCCESS){
				stringstream replyStringStream;
				replyStringStream << "Predicted label of image is: " << label;
				replyMessage = replyStringStream.str();
			}
			
			delete(request);
			break;
		}
		case SERV_REQ_HELP:
		{
			status = SERV_SUCCESS;
			stringstream replyStringStream;
			replyStringStream << endl <<"Usage: " << endl;
			replyStringStream << "train <input_filename.csv> <ouput_model_name.yml> " << endl;
			replyStringStream << "load  <model_name.yml> " << endl;
			replyStringStream << "unload  <model_name.yml> " << endl;
			replyStringStream << "recognize <input_image> <model_name.yml> " << endl;
			replyStringStream << "listloadedmodels " << endl;
			replyMessage = replyStringStream.str();
			break;
		}	
		case SERV_REQ_LIST_LOADED_MODELS:
		{
			string loaded_models;
			stringstream replyStringStream;
	
			status = ListLoadedModels(loaded_models);
			if(loaded_models.empty()){
				replyStringStream << endl <<"None" << endl;
			}
			else{
				replyStringStream << endl << loaded_models << endl;
			}
			replyMessage = replyStringStream.str();
			break;
		}
		default:
		{
			
			cout << "Process: Unrecognized request  "  << request->_requestType <<  endl;
   			std::cerr << "Process: Unrecognized request  "  << request->_requestType <<  endl;

			status = SERV_UNRECOGNIZED_COMMAND;

		}
	}


	return status;

}



/********************************************************************************************/
/*	enum_err_server FaceRecServer::parseClientRequest				    */
/*											    */
/*											    */
/********************************************************************************************/

enum_err_server FaceRecServer::parseClientRequest(char *clientReq, Request **reqParams)
{
	char req[MAX_CLIENT_COMMAND_LEN];
	char *request = NULL, *param1 = NULL, *param2 = NULL, *param3 = NULL;

	strncpy(req, clientReq, MAX_CLIENT_COMMAND_LEN);

	request = strtok( req," ");

	if(request == NULL){

		return 	SERV_UNRECOGNIZED_COMMAND;
	}	

	/*Request to train. Expect CSV file with images & labels*/
	if( !strcmp(request,"train") ){
		
		param1 = strtok(NULL," ");
		
		if(param1 == NULL){
			return SERV_CSV_FILE_NOT_SPEC;
		}

		cout << "parseClientRequest: train csv_filename  "  << param1 << endl;
   		std::cerr << "parseClientRequest: train csv_filename  "  << param1 << endl;

		param2 = strtok(NULL," ");

		if(param2 == NULL){
			return SERV_MODEL_NAME_NOT_SPEC;
		}

		cout << "parseClientRequest: train model_name  "  << param2 << endl;
   		std::cerr << "parseClientRequest: train model_name  "  << param2 << endl;

		*reqParams = new TrainRequest(param1, param2, SERV_REQ_TRAIN);

	}
	/*Request to load a model. Expect a model name*/
	else if( !strcmp(request,"load") ){
		
		param1 = strtok(NULL," ");
		
		if(param1 == NULL){

			return SERV_MODEL_NAME_NOT_SPEC;
		}

		*reqParams = new LoadModelRequest(param1, SERV_REQ_LOAD);

		cout << "parseClientRequest: load model_name  "  << param1 << endl;
   		std::cerr << "parseClientRequest: load model_name  "  << param1 << endl;
	
	}
	/*Request to recognize an image. Expect an input image name and model name*/
	else if( !strcmp(request,"recognize") ){

		param1 = strtok(NULL," ");
		
		if(param1 == NULL){

			return SERV_INPUT_IMAGE_NAME_NOT_SPEC;
		}

		cout << "parseClientRequest: recognize image  "  << param1 << endl;
   		std::cerr << "parseClientRequest: recognize image  "  << param1 << endl;

		param2 = strtok(NULL," ");
		
		if(param2 == NULL){

			return SERV_MODEL_NAME_NOT_SPEC;
		}


		cout << "parseClientRequest: recognize w/ model name  "  << param2 << endl;
   		std::cerr << "parseClientRequest: recognize w/ model name  "  << param2 << endl;

		*reqParams = new RecognizeRequest(param1,param2,SERV_REQ_RECOGNIZE);

	}
	/*Unload a model that has been loaded*/
	else if( !strcmp(request,"unload") ){

		param1 = strtok(NULL," ");
		
		if(param1 == NULL){

			return SERV_MODEL_NAME_NOT_SPEC;
		}

		cout << "parseClientRequest: unload  "  << param1 << endl;
   		std::cerr << "parseClientRequest: unload  "  << param1 << endl;

		*reqParams = new UnloadModelRequest(param1, SERV_REQ_UNLOAD);

	}
	/*Help Request*/
	else if( !strcmp(request,"help") ){
		cout << "parseClientRequest: help  "  <<  endl;
   		std::cerr << "parseClientRequest: help  "  << endl;

		*reqParams = new HelpRequest(SERV_REQ_HELP);	
	}
	/*List loaded Models*/
	else if( !strcmp(request,"listloadedmodels") ){

		cout << "parseClientRequest: list loaded models  "  <<  endl;
   		std::cerr << "parseClientRequest: list loaded model  "  << endl;

		*reqParams = new ListLoadedModelsRequest(SERV_REQ_LIST_LOADED_MODELS);
	}
	/*Unrecognized command*/
	else{
		return 	SERV_UNRECOGNIZED_COMMAND;
	}

	return SERV_SUCCESS;
}








