#include<stdio.h>
#include<stdlib.h>
#include<highgui.h>
#include<cv.h>
#include<algorithm>
#include<math.h>
#include<string>
#include<vector>
#include"getFeaturePoint.h"


using namespace std;

int main(){
    
    IplImage *img=0;
    vector<IplImage*> images;
	vector<feature> allFeatures;

    FILE *parameter=0;
    parameter=fopen("parameter.txt","r");
    
    
    int fileNumber=0;

	double sigma=0;
    int maskSize=0;
	int NMS_R=0;

    double f_length=0,s=0;

	int RANSAC_k=0,RANSAC_n=0;
	double RANSAC_t=0;

	char format[4];
	fscanf(parameter,"%s",format);
    fscanf(parameter,"%lf",&sigma);
    fscanf(parameter,"%s",&maskSize);
	fscanf(parameter,"%d",&NMS_R);
	fscanf(parameter,"%lf",&f_length);
	fscanf(parameter,"%lf",&s);
	fscanf(parameter,"%d",&RANSAC_k);
	fscanf(parameter,"%d",&RANSAC_n);
	fscanf(parameter,"%lf",&RANSAC_t);


    
    char file_to_open[10];
    
    
    sprintf(file_to_open,"%02d.%s",fileNumber,format);
    
    
    img=cvLoadImage(file_to_open);
    if(img==0){
        printf("error image\n");
        system("pause");
        return -1;
    }
    
    //get all feature
    while(img!=0){
        
        //cylindrical projection
        IplImage *cy=0;
        
		cy=cylindrical_projection(img,f_length,s);


        images.push_back(cy);
        
        //get feature

        allFeatures.push_back(getFeature(cy,sigma,maskSize,NMS_R));
    
        
        
        cvReleaseImage(&img);
        fileNumber++;
        fscanf(parameter,"%s",format);
        sprintf(file_to_open,"%02d.%s",fileNumber,format);
        img=0;
        img=cvLoadImage(file_to_open);
        
    }
    IplImage *end_to_head=0;
	end_to_head=cvCloneImage(images[0]);

	//remember to de comment
	//images.push_back(end_to_head);
	//allFeatures.push_back(allFratures[0]);
    
	
	cvNamedWindow("Stitching Result");
    
    IplImage *result=images[0];
    int modified_y=0;
    
	
    for(int i=0;i<images.size()-1;i++){
		
        IplImage *temp_result=0;
        temp_result = result;
        vector<pair<int,int>> match;
		match = featureMatching(allFeatures[i],allFeatures[i+1],images[i],images[i+1]);
		
		vector<CvPoint> L_matched,R_matched;
		for(int j=0;j<match.size();j++){
			L_matched.push_back(allFeatures[i].featureData[match[j].first].position);
			R_matched.push_back(allFeatures[i+1].featureData[match[j].second].position);
		}
		
		vector<int> trans;
		
        
		//RANSAC
		trans = RANSAC(L_matched,R_matched,images[i]->width,RANSAC_n,RANSAC_k,RANSAC_t);
		

        
        if(modified_y<0){
            modified_y=0;
        }
        modified_y +=trans[1];//要改成RANSAC找到的Y位移
		printf("\n X : %d \n",trans[0]);
		//Combine
        
		//result = connect_and_blend(temp_result,images[i+1],trans[0],modified_y);
		result = connect_and_blend(temp_result,images[i+1],trans[0],0);
		cvShowImage("Stitching Result",result);
		cvWaitKey(0);
		//cvReleaseImage(&images[i]);
		
    }
    
    cvShowImage("Stitching Result",result);
    //cvWaitKey(0);
	cvSaveImage("Result.jpg",result);
    fclose(parameter);

    
    return 0;
}