
#ifndef _FACE_REC_SERVER_H_

#define _FACE_REC_SERVER_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace std;
using namespace cv;

#define MAX_CLIENT_COMMAND_LEN (1024)	/*Max lenght for client request*/

#define PORT "3490"  /*the port client will connect to*/

#define BACKLOG 10   /*how many pending connections queue will hold*/

#define NUM_IMG_ROWS (243) /*Application will resize all images to this size*/

#define NUM_IMG_COLS (320) /*Application will resize all images to this size*/

/********************************************************************************************/
/*	Enum for Type of face recogntion algorithm					    */
/*											    */
/*											    */
/********************************************************************************************/

typedef enum
{
	FD_EIGEN_FACES = 1,
	FD_FISHER_FACES,
	FD_LBPH
}enum_face_detect_method;


/********************************************************************************************/
/*	Enum for Type request from client application recognized by server		    */
/*											    */
/*											    */
/********************************************************************************************/

typedef enum
{
	SERV_REQ_TRAIN = 0,
	SERV_REQ_LOAD,
	SERV_REQ_UNLOAD,
	SERV_REQ_RECOGNIZE,
	SERV_REQ_HELP,
	SERV_REQ_LISTMODELS,
	SERV_REQ_LIST_LOADED_MODELS,
}enum_serv_req;

/********************************************************************************************/
/*	Enum for Error codes returned by server						    */
/*											    */
/*											    */
/********************************************************************************************/

typedef enum
{
	SERV_SUCCESS = 0,
	SERV_UNRECOGNIZED_COMMAND,
	SERV_CSV_FILE_NOT_SPEC,
	SERV_MODEL_NAME_NOT_SPEC,
	SERV_INPUT_IMAGE_NAME_NOT_SPEC,
	SERV_ERR_PROCESSING_CSV,
	SERV_ERR_CREAT_MODEL,
	SERV_ERR_NO_IMAGES,
	SERV_ERR_CREATION_EIGEN_FAIL,
	SERV_ERR_FAIL_OPEN_STORAGE,
	SERV_ERR_MODEL_NOT_FOUND,
	SERV_ERR_FAIL_OPEN_IMG,
	SERV_ERR_LOADING_EIG_MODEL,
	SERV_ERR_FAIL_TRAIN_EIGEN,
	SERV_ERR_FAIL_SAVE_MODEL,
	SERV_ERR_RESIZE_IMG_FAIL,
	SERV_ERR_UNSUPP_FILE_FMT,
	SER_ERR_FAIL_OPEN_TMP_FILE,
	SERV_ERR_MAX

}enum_err_server;


/********************************************************************************************/
/*	Class: Request									    */
/*											    */
/*											    */
/********************************************************************************************/

class Request
{
public:
	Request(enum_serv_req reqType):_requestType(reqType)
	{
		_requestType = reqType;
	}
	enum_serv_req _requestType;

	virtual ~Request(){};
	
};

/********************************************************************************************/
/*	Class: TrainRequest								    */
/*											    */
/*											    */
/********************************************************************************************/

class TrainRequest:public Request
{

public:
	TrainRequest(string inputCSV, string outputModelName,enum_serv_req reqType = SERV_REQ_TRAIN):Request(reqType)
	{
	 	_inputCSVFileName = inputCSV;
		_outputModelName = outputModelName;
	}
	string _inputCSVFileName;
	string _outputModelName;
};

/********************************************************************************************/
/*	Class: LoadModelRequest								    */
/*											    */
/*											    */
/********************************************************************************************/

class LoadModelRequest:public Request
{

public:
	LoadModelRequest( string modelName,enum_serv_req reqType = SERV_REQ_LOAD):Request(reqType)
	{
	 	_modelName = modelName;
	}
	string _modelName;
};

/********************************************************************************************/
/*	Class: UnloadModelRequest								    */
/*											    */
/*											    */
/********************************************************************************************/

class UnloadModelRequest:public Request
{

public:
	UnloadModelRequest( string modelName,enum_serv_req reqType = SERV_REQ_UNLOAD):Request(reqType)
	{
	 	_modelName = modelName;
	}
	string _modelName;
};

/********************************************************************************************/
/*	Class: RecognizeRequest								    */
/*											    */
/*											    */
/********************************************************************************************/
class RecognizeRequest:public Request
{

public:
	RecognizeRequest( string inputImage,string modelName, enum_serv_req reqType = SERV_REQ_RECOGNIZE):Request(reqType)
	{
	 	_modelName = modelName;
		_inputImage = inputImage;
	}
	string _modelName;
	string _inputImage;
};

/********************************************************************************************/
/*	Class: HelpRequest								    */
/*											    */
/*											    */
/********************************************************************************************/
class HelpRequest:public Request
{

public:
	HelpRequest( enum_serv_req reqType = SERV_REQ_HELP):Request(reqType)
	{
		//nothing
	}

};

/********************************************************************************************/
/*	Class: ListLoadedModelsRequest								    */
/*											    */
/*											    */
/********************************************************************************************/
class ListLoadedModelsRequest:public Request
{

public:
	ListLoadedModelsRequest( enum_serv_req reqType = SERV_REQ_LIST_LOADED_MODELS):Request(reqType)
	{
		//nothing
	}

};

/********************************************************************************************/
/*	Class: FaceRecModel								    */
/*											    */
/*											    */
/********************************************************************************************/


class FaceRecModel
{
public:
	FaceRecModel(string modelName)
	{
		_modelName = modelName;
	}


	string _modelName;
	vector<Mat> _images;
	vector<int> _labels;
	vector<string> _persons;

	/*OpenCV core model*/
	Ptr<FaceRecognizer> model_core;

	~FaceRecModel()
	{
		cout << "Face recognition model destructor called for "  << _modelName << endl;
   		std::cerr << "Face recognition model destructor called "  << _modelName << endl;
		//delete &model_core;
	}
	bool operator ==(const FaceRecModel &mdl) const {
		return _modelName == mdl._modelName;
	}

};

/********************************************************************************************/
/*	Class: FaceRecServer								    */
/*											    */
/*											    */
/********************************************************************************************/

class FaceRecServer
{

public:
	FaceRecServer();

	FaceRecServer(int maxModels){ }

	enum_err_server TrainModel(string filenamesCsv, string modelName, enum_face_detect_method algorithm = FD_EIGEN_FACES);

	enum_err_server LoadModel(string modelName, bool reload = false);

	enum_err_server UnloadModel(string modelName);

	enum_err_server recognizeImage(string inputImage, string modelName, int &label );

	enum_err_server parseClientRequest(char *request, Request **reqParams = NULL);

	enum_err_server ProcessRequest(Request *request, string &replyMessage);

	FaceRecModel& searchModelsByModelName(string modelName);

	enum_err_server ListLoadedModels(string &loaded_models);

	
private:	
	vector<FaceRecModel *> models;
	
};


#endif /*_FACE_REC_SERVER_H_*/
