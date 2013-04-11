/*
 Code for the Science Channel Demo of :
 Griffith. Sinapov. Miller. and Stoytchev. ``Toward Interactive Learning of Object Categories by a Robot: A Case Study with Containers and Non-Containers.''
 ICDL. 2009
 
 @author: Shane Griffith. June 2009.
 
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include <tdvCameraInterface.h>
#include "DepthCamera.h"
#include <cv.h>						//vision processing
#include <highgui.h>

#include <Mmsystem.h>
#include <conio.h>
#include <windows.h>
#include <winbase.h>
#include <stdio.h>

#include "stdafx.h"
#include "DataWriter.h"
#include "SocketComm.h"				//feature detection (matlab) and action (robot)
#include "ImageTypeWrapper.h"
#include "voce.h"					//speech recognition

#include <signal.h>

#define SLEEPTIME 30
#define BUFSIZE 1024
#define WAIT true
#define NO_WAIT false
#define	TERMINATE -1
#define CORRECT 1
#define INCORRECT 0
#define NOTHING_HEARD -2

char rootname[30] = "C:/exp/";
IplImage * depthmax;
IplImage * depthmin;
IplImage * colormax;
IplImage * colormin;
int imgnum;

/** The background has a certain color and depth. Acquire this model.
 *
 *
 */
void AcquireBackgroundModel(IplImage * colorImg, IplImage * depthImg)
{
	RgbImage _ccur(colorImg);
	RgbImage _cmax(colormax);
	RgbImage _cmin(colormin);
	
	BwImage _dcur(depthImg);
	BwImage _dmax(depthmax);
	BwImage _dmin(depthmin);
	printf("beg of function.\n");

	for(int i=0; i<colorImg->height; i++)
	{
		for(int j=0; j<colorImg->width; j++)
		{
			//depth min, max
			if(_dcur[i][j] < _dmin[i][j]) _dmin[i][j] = _dcur[i][j];
			if(_dcur[i][j] > _dmax[i][j]) _dmax[i][j] = _dcur[i][j];
			
			//color min, max
			if(_ccur[i][j].b < _cmin[i][j].b) _cmin[i][j].b = _ccur[i][j].b;
			if(_ccur[i][j].b > _cmax[i][j].b) _cmax[i][j].b = _ccur[i][j].b;
			if(_ccur[i][j].g < _cmin[i][j].g) _cmin[i][j].g = _ccur[i][j].g;
			if(_ccur[i][j].g > _cmax[i][j].g) _cmax[i][j].g = _ccur[i][j].g;
			if(_ccur[i][j].r < _cmin[i][j].r) _cmin[i][j].r = _ccur[i][j].r;
			if(_ccur[i][j].r > _cmax[i][j].r) _cmax[i][j].r = _ccur[i][j].r;
		}
	}
}

/** Mask pixels that don't fall within the color range and depth range.
 *
 *
 */
void MaskBackgroundPixels(IplImage * colorImg, IplImage * depthImg, IplImage * mask, IplConvKernel* element)
{
	int SIGMA = 10;
	RgbImage _ccur(colorImg);
	BwImage _dcur(depthImg);
	RgbImage _cmax(colormax);
	RgbImage _cmin(colormin);
	BwImage _dmax(depthmax);
	BwImage _dmin(depthmin);
	BwImage _mask(mask);

	for(int i=0; i<colorImg->height; i++)
	{
		for(int j=0; j<colorImg->width; j++)
		{
			int matches = 0;
			
			matches += (_ccur[i][j].b+SIGMA >= _cmin[i][j].b)?1:0;
			matches += (_ccur[i][j].b <= _cmax[i][j].b+SIGMA)?1:0;
			matches += (_ccur[i][j].g+SIGMA >= _cmin[i][j].g)?1:0;
			matches += (_ccur[i][j].g <= _cmax[i][j].g+SIGMA)?1:0;
			matches += (_ccur[i][j].r+SIGMA >= _cmin[i][j].r)?1:0;
			matches += (_ccur[i][j].r <= _cmax[i][j].r+SIGMA)?1:0;

			if(matches >= 5)
			{
				_mask[i][j] = 0;
			}
			else
			{
				_mask[i][j] = 255;
			}
		}
	}

	cvErode(mask,mask,element,2);
	cvDilate(mask,mask,element,5); //leave an extra dilation so the outer rim of the objects aren't cut away
	cvErode(mask,mask,element,2);
}

CvRect GetBoundingBox(IplImage * mask, IplImage * color)
{
  //find the largest blob, then get the top left position
  int minThreshold = 3000;
  int maxsize = -1;
  CvSeq* MaxContour = 0;

  //declare variables for the foreground contour objects
  CvMemStorage* storage = cvCreateMemStorage(0);
  CvSeq* contour = 0;
  CvContourScanner traverse = NULL;
  CvScalar ret = cvScalar(-1, -1, -1, -1);
  
  //create a contour iteration
  traverse = cvStartFindContours(mask, storage, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
  
  //get the contour of the next iteration
  contour = cvFindNextContour(traverse);
  
  //find contours within minThreshold and maxThreshold lengths
  while(contour != NULL){
    //only use contours falling within the threshold values
    int contourarea = (int) abs(cvContourArea(contour,CV_WHOLE_SEQ));

	//find the largest blob that fits within the specified range
	if(contourarea > maxsize && contourarea > minThreshold)
	{
		maxsize = contourarea;
		MaxContour = contour;
		CvRect rec = cvBoundingRect(contour,0);
		cvRectangle(color, cvPoint(rec.x, rec.y), cvPoint(rec.x+rec.width, rec.y+rec.height), cvScalar(0,0,255), 2, 8, 0);
	}
	
    contour = cvFindNextContour(traverse);
  }

  if(maxsize > 0)
  {
	  //cut off the top region of the bottle (since a full bottle goes about 75% of the way up.
	  return cvBoundingRect(MaxContour,0);
  }
  
  //release memory
  contour = cvEndFindContours(&traverse);
  cvReleaseMemStorage(&storage);
  return cvRect(-1, -1, -1, -1);
}

void Get30x30Image(IplImage * depth, CvRect rec, IplImage * vector)
{
	cvSetImageROI(depth, rec);
	cvResize(depth, vector, CV_INTER_LINEAR); //CV_INTER_AREA);
	cvResetImageROI(depth);
}

void SaveImage(IplImage * img)
{
	static int imgnum = 0;
	char filename[100] = "D:/school/Vision Research/Code/Sparse Coding/HRItest/img1.jpg";
	cvSaveImage(filename, img);
	sprintf(filename, "c:/HRIExperiment/img%d.jpg", imgnum);
	cvSaveImage(filename, img);
	imgnum++;
}

/**write the string in 'str' to the file specified
 */
void writeToFile(char filename[], char str[])
{
     FILE *fp = NULL;
     
     //if the file does not exist yet, it is created and opened for writing.
     //if the file exists, the contents of the current file are erased and 
     //overwritten with the contents of 'str'
     fp = fopen(filename, "w");
     fputs(str, fp);
     fclose(fp);
}

void appendToFile(char filename[], char str[])
{
     FILE *fp = NULL;
     
     //if the file does not exist yet, it is created and opened for writing.
     //if the file exists, the contents of 'str' are appended to the end
     fp = fopen(filename, "a");
     fputs(str, fp);
     fclose(fp);
}

void AddTestImageToTrainingSet(IplImage * vectorImg, int imglabel)
{
	char textlbl[10];
	char to[100];
	sprintf(textlbl, ",%d", imglabel);

	//save the test image in the training set
	sprintf(to, "D:/school/Vision Research/Code/Sparse Coding/HRItrain/img%d.jpg", imgnum); 
	cvSaveImage(to, vectorImg);

	//append the classification to the text file.
	appendToFile("D:/school/Vision Research/Code/Sparse Coding/HRItrain/labels.csv", textlbl);
	
	//move image
	//CopyFile("D:/school/Vision Research/Code/Sparse Coding/HRItest/img1.jpg", to);
	imgnum++;
}

int SendMessage(int socket, char * msg)
{
	const void * x = msg;
	
	if(safewrite(socket, x, 2) != 2)
	{
		printf("Problem communicating with Matlab. Quitting.");
		return 0;
	}
	return 1;
}

char GetMatlabClassification(int matlabSocket)
{
	char buf[1];
	//send command to matlab to analyze the image
	char msg[10] = "51";
	if(SendMessage(matlabSocket, msg) == 0) return '\0';

	printf("waiting for reply\n");
	//wait for finish reply from matlab
	while(1)
	{
		if(CheckRxBuffer(matlabSocket))
		{
			if(saferead(matlabSocket, buf, 1) != 1)
			{
				printf("Could not read the packet.\n");
			}

			switch(buf[0])
			{
			case 'n':printf("non-container.\n"); return 'n';
			case 'c':printf("container.\n"); return 'c';
			case '?':printf("unknown.\n"); return '?';
			default: printf("Received  %c  Unacceptable classification.\n", buf[0]); return '\0';
			}
		}
		else
		{
			Sleep(SLEEPTIME);
		}
	}
}

int SendCommandToRobot(int robotSocket, char cmd)
{
	if(safewrite(robotSocket, &cmd, 1) != 1)
	{
		printf("Could not communicate with the Robot. Quitting.");
		return 0;
	}
	return 1;
}

int isActionFinished(int robotSocket)
{
	static int count = 150;  //7.5 seconds
	
	if(count==0)
	{
		printf("Robot finished moving. \n");
		count = 50;
		printf("\n\nClear the board.\n\n");
		system("pause");
		return 1;
	}
	Sleep(50);

	count--;
	return 0;
}	

char GetRobotReply(int robotSocket)
{
	char buf[1];
	printf("Waiting for the robot to finish the action.\n");
	
	//wait a minute for the robot to complete its action.
	for(int i=0; i<2000; i++)
	{
		if(CheckRxBuffer(robotSocket))
		{
			//any response is a good response.
			if(saferead(robotSocket, buf, 1) != 1)
			{
				printf("Could not read the packet.\n");
			}
			else
			{
				return buf[0];
			}
		}
		else
		{
			Sleep(SLEEPTIME);
		}
	}
	return '\0';
}

int GetFeedback(char c, bool wait)
{
	int numPos = 0;
	bool heardClassification = false;
	
	if(wait)
	{
		while(!heardClassification)
		{
			if(voce::getRecognizerQueueSize() > 0)
			{
				std::string s = voce::popRecognizedString();
				std::cout << "You said: " << s << std::endl;
				if(s.compare("no")==0)
				{
					numPos++;
					heardClassification = true;
				}
				else if(s.compare("yes")==0)
				{
					numPos--;
					heardClassification = true;
				}
				else if(s.compare("quit")==0) return TERMINATE;
			}
			
			Sleep(200);
		}
	}

	//check for input and clear the speech recognition queue
	while (voce::getRecognizerQueueSize() > 0)
	{
		std::string s = voce::popRecognizedString();
		std::cout << "You said: " << s << std::endl;
		if(s.compare("no")==0)
		{
			printf("\t\theard NO\n");
			numPos++;
			heardClassification = true;
		}
		else if(s.compare("yes")==0)
		{
			printf("\t\theard YES\n");
			numPos--;
			heardClassification = true;
		}
		else if(s.compare("quit")==0) return TERMINATE;
	}
	
	if(heardClassification == true)
	{
		return (numPos>0)?INCORRECT:CORRECT;
	}
	else 
		return NOTHING_HEARD;
}

char CheckWithUserInput(char c)
{
	char user;
	printf("What is the classification? (c or n)?\n");
	scanf("%c", &user);
	getchar();
	if(c != user) return '0';
	else return '1';
}

int main(int argc, char *argv[])
{
	unsigned char * pDepth = NULL;
	unsigned char * pRGB = NULL;
	unsigned char * pRGBFullRes = NULL;
	unsigned char * pPrimary = NULL;
	unsigned char * pSecondary = NULL;
	int GetNextImage = 200;
	int toMatlab = -1, toRobot = -1;
	int client=-1;
	bool record = false;
	char buf[200];
	char imageFilename[150];
	char depthFilename[150];
	int trial=0;
	int width, height, dSize, rgbSize, rgbFullResWidth, rgbFullResHeight;
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);

	IplConvKernel* element = cvCreateStructuringElementEx( 3, 3, 1, 1, CV_SHAPE_RECT, 0 );

	/**************** CAMERA SYSTEM ****************/
	CDepthCamera depthCamera;
	if (!depthCamera.Initialize())
	{
		printf("Unable to initialize camera!... exiting...\n");
		return 0;
	}
	printf("Camera is connected!\n");

	// set default values for a selected depth range (distance=65 and width = 150, 
	// meaning a depth ranging from 65cm up to 215cm)
	depthCamera.SetDepthWindowPosition(170, 100);  //204, 91
	depthCamera.GetVideoSize(width, height, rgbFullResWidth, rgbFullResHeight, dSize, rgbSize);
	TDVCameraInterfaceBase * x = depthCamera.GetCameraInterface();
	x->cmdRestoreStatus(0);

	/**************** ROBOT SYSTEM ****************/
	toRobot = createClientSocket("3000", "129.186.159.117");
	if(toRobot == -1)
	{
		printf("Could not connect to Robot.\n");
		exit(1);
	}
	printf("Connected to Robot\n", toRobot);

	/**************** FEATURE EXTRACTION AND LEARNING SYSTEM ****************/
	toMatlab = createClientSocket("3003", "localhost");
	if(toMatlab == -1)
	{
		printf("Could not connect to Matlab.\n");
		exit(1);
	}
	printf("Connected to Matlab\n", toMatlab);
	
	/**************** SOUND SYSTEM ****************/
	//voce::init("D:/school/Vision Research/Code/voce/lib", false, true, "./grammar", "classification");



	//initialize the label file
	writeToFile("D:/school/Vision Research/Code/Sparse Coding/HRItrain/labels.csv", "1,0,1,0,1,0,1,0,1,0");

	Sleep(500); //wait for the ZCam settings to take effect, and give enough time for user to see status

	//images to hold data from the camera
	IplImage* colorImgTEMP = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 4);
	IplImage* colorImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	IplImage* hsvImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	IplImage* depthImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	IplImage* mask = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	IplImage* mask2 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	IplImage* vectorImg = cvCreateImage(cvSize(30, 30), IPL_DEPTH_8U, 1);

	//for cvsplit function
	IplImage* temp1 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	IplImage* temp2 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	IplImage* temp3 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	IplImage* junk = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);

	depthmax = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	depthmin = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	colormax = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 4);
	colormin = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 4);

	cvSet(mask,cvScalarAll(255),0);
	cvSet(depthmax,cvScalarAll(0),0);
	cvSet(depthmin,cvScalarAll(255),0);
	cvSet(colormax,cvScalarAll(0),0);
	cvSet(colormin,cvScalarAll(255),0);

	//create the background model (should take >= 5 seconds)
	for(int i=0; i<150; i++)
	{
		depthCamera.GetNextFrame(pDepth, pRGB, pRGBFullRes, pPrimary, pSecondary);
		memcpy(colorImgTEMP->imageData, pRGB, width*height*rgbSize);
		memcpy(depthImg->imageData, pDepth, width*height*dSize);
		cvCvtPixToPlane(colorImgTEMP, temp1, temp2, temp3, junk);
		cvCvtPlaneToPix(temp1, temp2, temp3, 0, colorImg);
		cvCvtColor(colorImg, hsvImg, CV_BGR2HSV);

		printf("%d, ", i);
		AcquireBackgroundModel(hsvImg, depthImg);
	}
	
	printf("\n\nDone creating background model.\n\n");

	int numObjects = 10;
	char command;
	char input;
	cvNamedWindow("Robot Field Of View", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Extracted Feature", CV_WINDOW_AUTOSIZE);
	for(int i=0; i<numObjects; i++)
	{
		//wait for input from the user before continuing.
		printf("Input _c_ when the next object is on the table.\n");
		while(input != 'c')
		{
			if(CheckRxBuffer(fileno(stdin)))
			{
				scanf("%c\n", &input);
			}
			depthCamera.GetNextFrame(pDepth, pRGB, pRGBFullRes, pPrimary, pSecondary);
			memcpy(colorImgTEMP->imageData, pRGB, width*height*rgbSize);
			memcpy(depthImg->imageData, pDepth, width*height*dSize);
			cvCvtPixToPlane(colorImgTEMP, temp1, temp2, temp3, junk);
			cvCvtPlaneToPix(temp1, temp2, temp3, 0, colorImg);
			cvShowImage("Robot Field Of View", colorImg);
			cvWaitKey(30);
		}

		//get model of the object.
		cvCvtColor(colorImg, hsvImg, CV_BGR2HSV);
		
		MaskBackgroundPixels(hsvImg, depthImg, mask, element);

		cvCopy(mask, mask2, 0);		//save a copy of the depth image to show (debugging)

		CvRect box = GetBoundingBox(mask, colorImg);
		cvShowImage("depth", mask2);
		cvShowImage("color", colorImg);
		cvWaitKey(30);
		
		Get30x30Image(depthImg, box, vectorImg);

		//send command to the robot to interact with it.
		command = 'g';
		SendCommandToRobot(toRobot, command); //tell robot to perform its interaction.	

		//display the image.
		cvShowImage("Extracted Feature", vectorImg);
		cvWaitKey(30);
	}

	//find new objects and check for input
	bool continueOperation = true;
	bool havePerceptualModel = true;
	imgnum = 11; //images start saving with number _imgnum_
	while(continueOperation)
	{
		//wait for input from the user before continuing.
		printf("Input _c_ when the next object is on the table.\n");
		while(input != 'c')
		{
			if(CheckRxBuffer(fileno(stdin)))
			{
				scanf("%c\n", &input);
			}
			depthCamera.GetNextFrame(pDepth, pRGB, pRGBFullRes, pPrimary, pSecondary);
			memcpy(colorImgTEMP->imageData, pRGB, width*height*rgbSize);
			memcpy(depthImg->imageData, pDepth, width*height*dSize);
			cvCvtPixToPlane(colorImgTEMP, temp1, temp2, temp3, junk);
			cvCvtPlaneToPix(temp1, temp2, temp3, 0, colorImg);
			cvShowImage("Robot Field Of View", colorImg);
			cvWaitKey(30);
		}

		//get model of the object.
		cvCvtColor(colorImg, hsvImg, CV_BGR2HSV);
		
		MaskBackgroundPixels(hsvImg, depthImg, mask, element);

		cvCopy(mask, mask2, 0);		//save a copy of the depth image to show (debugging)

		CvRect box = GetBoundingBox(mask, colorImg);
		cvShowImage("depth", mask2);
		cvShowImage("color", colorImg);
		cvWaitKey(30);
		
		Get30x30Image(depthImg, box, vectorImg);
		cvShowImage("Extracted Feature", vectorImg);

		SaveImage(vectorImg);
		
		printf("Querying matlab for classification.\n");
		char classification = GetMatlabClassification(toMatlab);
		
		switch(classification)
		{
			case '\0': 
				continueOperation = false;  //problem. Terminate execution
				continue;
			case '?': 
				printf("Received question mark...???\n");
				getchar();
				getchar();
				break;
			case 'c':
				//write text on the image to say 'container'
				cvPutText(colorImg, "container", cvPoint(50, 200), &font, cvScalar(255, 255, 255, 0));
			case 'n':
				//write text on the image to say 'non-container'
				cvPutText(colorImg, "non-container", cvPoint(50, 200), &font, cvScalar(255, 255, 255, 0));
				break;
		}
		
		cvShowImage("color", colorImg);
		cvWaitKey(0);
	}

	SendMessage(toMatlab, "50");		//terminate the HRI program
	command = 'T';
	//SendCommandToRobot(toRobot, command);			//terminate the robot
	Sleep(1000);						//don't destroy anything before sending the stop command.
	closesocket(toMatlab);
	closesocket(toRobot);
	cvDestroyWindow("depth");
	cvDestroyWindow("color");
	cvDestroyWindow("Extracted Feature");
	cvDestroyWindow("Robot Field Of View");
}






