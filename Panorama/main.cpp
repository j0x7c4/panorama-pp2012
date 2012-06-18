#include<stdio.h>
#include<stdlib.h>
#include<highgui.h>
#include<imgproc\imgproc.hpp>
#include<cv.h>
#include<algorithm>
#include<math.h>
#include<string>
#include<vector>
#include"getFeaturePoint.h"
#include<omp.h>
#include<advisor-annotate.h>

using namespace std;
//#define PARALLEL
const char* source_dir = "D:\\source\\Panorama\\Debug\\source";
const int norm_height = 600;

int main(){

  IplImage *temp_img=NULL;
  vector<IplImage*> images;
  vector<feature> allFeatures;
  vector<string> files_to_open;
  FILE *parameter=NULL;
  parameter=fopen("parameter.txt","r");


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
  fclose(parameter);


  char file_to_open[255];

  //img=cvLoadImage(file_to_open);

  //load pictures in dir
  HANDLE hFind;
  WIN32_FIND_DATAA FindFileData;
  char pattern[255];
  sprintf(pattern,"%s\\*.%s",source_dir,format);
  if((hFind = FindFirstFileA(pattern, &FindFileData)) != INVALID_HANDLE_VALUE){
    do{
      sprintf(file_to_open,"%s\\%s",source_dir,FindFileData.cFileName);
      files_to_open.push_back(string(file_to_open));
    }while(FindNextFileA(hFind, &FindFileData));
    FindClose(hFind);
  }
  else {
    printf ("FindFirstFile failed (%d)\n", GetLastError());
    return 1;
  }

  clock_t time_start = clock();

  //openmp need "for", change while to for
  images.resize(files_to_open.size(),NULL);
  allFeatures.resize(files_to_open.size(),feature());
#ifdef PARALLEL
#pragma omp parallel for private(temp_img)
#endif
  for ( int i = 0 ; i<files_to_open.size() ; i++ ) {
    printf("Loading %s...\n",files_to_open[i].c_str());
    //get feature
    temp_img=cvLoadImage(files_to_open[i].c_str());
    //down sample
    int width = temp_img->width, height = temp_img->height;
    while ( width > 1000 || height > 1000 ) {
      width/=2;
      height/=2;
    }
    IplImage* down_img = cvCreateImage(cvSize( width, height),temp_img->depth,temp_img->nChannels);
    cvResize(temp_img, down_img);
    //cylindrical projection
    IplImage *cy=0;
    cy = cylindrical_projection(down_img,f_length,s);
    images[i] = cy;
    //get feature
    allFeatures[i] = getFeature(cy,sigma,maskSize,NMS_R);
    cvReleaseImage(&temp_img);
    cvReleaseImage(&down_img);
  }

  //remember to de comment
  //images.push_back(end_to_head);
  //allFeatures.push_back(allFratures[0]);


  //cvNamedWindow("Stitching Result",CV_WINDOW_AUTOSIZE);
  IplImage *result=images[0];

  vector<vector<int>> trans(images.size()-1,vector<int>());
  
  int modified_y=0;
  int ret_height=images[0]->height,ret_width=images[0]->width;
  vector<pair<int,int>> offset(images.size()+1,pair<int,int>(0,0));
  for(int i=0;i<images.size()-1;i++){
    vector<CvPoint> L_matched,R_matched;
    vector<pair<int,int>> match;
    match = featureMatching(allFeatures[i],allFeatures[i+1],images[i],images[i+1]);

    for(int j=0;j<match.size();j++){
      L_matched.push_back(allFeatures[i].featureData[match[j].first].position);
      R_matched.push_back(allFeatures[i+1].featureData[match[j].second].position);
    }
    
    //RANSAC
    trans[i] = RANSAC(L_matched,R_matched,images[i]->width,RANSAC_n,RANSAC_k,RANSAC_t);
    if(modified_y<0){
      modified_y=0;
    }
    modified_y += trans[i][1];//要改成RANSAC找到的Y位移
    offset[i+1].first=offset[i].first+images[i]->width-trans[i][0];
    offset[i+1].second=0;

    ret_width+=images[i+1]->width-trans[i][0];
    ret_height = max(ret_height,images[i+1]->height);

  }
  offset[images.size()].first=ret_width;
  offset[images.size()].second=0;

  IplImage* ret = cvCreateImage(cvSize(ret_width,ret_height),8,3);
  cvSetZero(ret);
  unsigned char* dst_data = (unsigned char*)ret->imageData;
#ifdef PARALLEL
#pragma omp parallel for
#endif
  for ( int i=0 ; i<images.size() ; i++ ) {
    if ( i==0 ) {
      for ( int y = 0 ; y<images[i]->height ; y++ ) {
        unsigned char* dst_row_pointer = dst_data+y*ret->widthStep;
        unsigned char* src_row_pointer = (unsigned char*)images[0]->imageData+y*images[0]->widthStep;
        for ( int x = 0 ; x<offset[1].first ; x++ ) {
          *(dst_row_pointer+3*x)=*src_row_pointer++;
          *(dst_row_pointer+3*x+1)=*src_row_pointer++;
          *(dst_row_pointer+3*x+2)=*src_row_pointer++;
        }
      }
    }
    
    else {
      parallel_connect_and_blend(images[i-1],images[i],
                                 cvRect(offset[i].first,offset[i].second,offset[i+1].first-offset[i].first,images[i]->height),
                                 cvRect(offset[i].first,offset[i].second,images[i-1]->width+offset[i-1].first-offset[i].first,images[i]->height),
                                 ret_height,
                                 ret_width,
                                 dst_data,
                                 ret->widthStep);
                                                        
    }
    //cvShowImage("Stitching Result",ret);
    //cvWaitKey(0);
  }


  printf("Time\t%lf\n",(clock()-time_start)/1000.0);
  //printf("%d %d\n",ret_height,ret_width);
  //printf("%d %d\n",result->height,result->width);
  //cvShowImage("Stitching Result",result);
  //cvWaitKey(0);
  cvSaveImage("result.jpg",ret);



  return 0;
}