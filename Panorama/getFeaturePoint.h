//with discriptor return #
#ifndef GETFEATUREPOINT_H
#define GETFEATUREPOINT_H

#include<stdio.h>
#include<stdlib.h>
#include<highgui.h>
#include<cv.h>
#include<algorithm>
#include<math.h>
#include<string>
#include<vector>
#include<ANN.h>
#include<time.h>
#include<omp.h>

using namespace std;

typedef struct onePoint{
  CvPoint position;
  vector<double> discriptor;
}featurePoint;

typedef struct imgFeature{
  vector<featurePoint> featureData;
}feature;



//void computeGradient(IplImage *src,CvMat *dst,int direction);
void computeGradient(IplImage *src,CvMat *dst1,CvMat *dst2,CvMat *dst3);
void check_local(CvMat *src,CvMat *dst);
void NMS(CvMat *src,CvMat *dst,int num,int R);
IplImage* connect_and_blend(IplImage *left_src,IplImage *right_src,int cols,int y);
feature saveDiscriptor(IplImage *src,CvMat *point,CvMat *xG,CvMat *yG);
featurePoint saveOnePointDiscriptor(IplImage *src,int x,int y,double angle);

vector<double> getDiscriptor(IplImage *src,int x,int y,double angle);
feature getFeature(IplImage *src,double sigma,int masksize,int R);

//projection
IplImage* cylindrical_projection(IplImage *src,double f,double s);
void reconstruction(IplImage *src,double x, double y,IplImage *dst,int row,int column);
vector<int> RANSAC(vector<CvPoint> left,vector<CvPoint> right,int width,int n,int k,double t);
vector<int> Alignment(vector<CvPoint> left,vector<CvPoint> right,int width);

feature getFeature(IplImage *src,double sigma,int masksize,int R){
  int H = src->height,W = src->width;
  IplImage *img=0;
  img=cvCreateImage(cvSize(W,H),IPL_DEPTH_8U,1);
  cvCvtColor(src,img,CV_BGR2GRAY);

  IplImage *img2=0;
  img2=cvCreateImage(cvSize(W,H),IPL_DEPTH_8U,3);
  img2=cvCloneImage(src);



  IplImage *img_s=0;
  img_s = cvCreateImage(cvSize(W,H),IPL_DEPTH_8U,1);
  CvMat ImageData;

  cvSmooth(img,img_s,CV_GAUSSIAN,masksize,0,1.0);

  CvMat *X_gradient ,*Y_gradient, *XY_gradient,*X_gradient_for_dis,*Y_gradient_for_dis;
  X_gradient = cvCreateMat(H,W,CV_32FC1); 
  Y_gradient = cvCreateMat(H,W,CV_32FC1);
  XY_gradient = cvCreateMat(H,W,CV_32FC1);

  X_gradient_for_dis = cvCreateMat(H,W,CV_32FC1); 
  Y_gradient_for_dis = cvCreateMat(H,W,CV_32FC1);


  computeGradient(img_s,X_gradient,Y_gradient,XY_gradient);   

  cvSmooth(X_gradient,X_gradient_for_dis,CV_GAUSSIAN,masksize,0,4.5);
  cvSmooth(Y_gradient,Y_gradient_for_dis,CV_GAUSSIAN,masksize,0,4.5);
  cvSmooth(X_gradient,X_gradient,CV_GAUSSIAN,masksize,0,sigma);
  cvSmooth(Y_gradient,Y_gradient,CV_GAUSSIAN,masksize,0,sigma);
  cvSmooth(XY_gradient,XY_gradient,CV_GAUSSIAN,masksize,0,sigma);

  CvMat *F_MAP=0;
  F_MAP = cvCreateMat(H,W,CV_32FC1);

  CvMat *e_val=0,*e_vec=0,*DT=0,*L_max=0; 

  DT = cvCreateMat(2,2,CV_32FC1);
  e_val = cvCreateMat(2,1,CV_32FC1);
  e_vec = cvCreateMat(2,2,CV_32FC1);
  L_max = cvCreateMat(H,W,CV_32FC1);

  IplImage *test=0;
  test = cvCreateImage(cvSize(W,H),IPL_DEPTH_8U,1);
  float* DT_data = DT->data.fl;
  float* X_gradient_data = X_gradient->data.fl;
  float* XY_gradient_data = XY_gradient->data.fl;
  float* Y_gradient_data = Y_gradient->data.fl;
  float* F_MAP_data = F_MAP->data.fl;
  float* L_max_data = L_max->data.fl;
  for(int i=0;i<H;i++){
    for(int j=0;j<W;j++){
      DT_data[0]=X_gradient_data[i*W+j]; 
      DT_data[1]=XY_gradient_data[i*W+j];
      DT_data[2]=Y_gradient_data[i*W+j];
      DT_data[3]=XY_gradient_data[i*W+j];
      cvEigenVV(DT,e_vec,e_val,DBL_EPSILON);
      double thre=(e_val->data.fl[0]* e_val->data.fl[1])-0.04*(e_val->data.fl[0] + e_val->data.fl[1]);
      //double thre=(e_val->data.fl[0]* e_val->data.fl[1])/(e_val->data.fl[0] + e_val->data.fl[1]);
      F_MAP_data[i*W+j] = thre;
      L_max_data[i*W+j] = 0;
    }
  }

  check_local(F_MAP,L_max);


  CvMat *F_MAP_LM=0;
  F_MAP_LM = cvCreateMat(img->height,img->width,CV_32FC1);

  CvMat *F_MAP_LM_NMS=0;
  F_MAP_LM_NMS = cvCreateMat(img->height,img->width,CV_32FC1);

 
  for(int i=0;i<img->height;i++){
    for(int j=0;j<img->width;j++){
      if(L_max->data.fl[i*L_max->cols+j]>0 && F_MAP->data.fl[i*L_max->cols+j]>10){
        F_MAP_LM->data.fl[i*F_MAP_LM->cols+j]=F_MAP->data.fl[i*L_max->cols+j];
      }
      else{
        F_MAP_LM->data.fl[i*F_MAP_LM->cols+j]=0;
      }
    }
  }



  NMS(F_MAP_LM,F_MAP_LM_NMS,500,R);


  double max=0;
  int max_r=0,max_c=0;
 
  for(int i=0;i<img->height;i++){
    for(int j=0;j<img->width;j++){
      if(F_MAP_LM_NMS->data.fl[i*F_MAP_LM_NMS->cols+j]>0){
        test->imageData[i*test->widthStep+j]=255;
        cvCircle(img2,cvPoint(j,i),4,cvScalar(0,0,255),1,8,0);
      }
      else{
        test->imageData[i*test->widthStep+j]=0;
      }
    }
  }


  feature data;

  data = saveDiscriptor(img,F_MAP_LM_NMS,X_gradient_for_dis,Y_gradient_for_dis);

  //   cvNamedWindow("C");
  //   cvShowImage("C",test);
  //cvSaveImage("test_color_H333.bmp",img2);
  //cvWaitKey(0);
  //  cvReleaseImage(&img); 

  return data;

}

void computeGradient(IplImage *src,CvMat *dst1,CvMat *dst2,CvMat *dst3){
  double gradient_x;
  double x1,x2;
 
  for(int i=0;i<src->height;i++){
    for(int j=0;j<src->width;j++){
      if(j==0){
        x1=(double)((unsigned char)src->imageData[i*src->widthStep]);
      }
      else{
        x1=(double)((unsigned char)src->imageData[i*src->widthStep+(j-1)]);
      }

      if(j==src->width-1){
        x2=(double)((unsigned char)src->imageData[i*src->widthStep+j]);
      }
      else{
        x2=(double)((unsigned char)src->imageData[i*src->widthStep+(j+1)]);
      }
      gradient_x = (x2-x1)/2;
      dst1->data.fl[i*dst1->cols+j]=gradient_x;
    }
  }

  double gradient_y;
  double y1,y2;
 
  for(int i=0;i<src->height;i++){
    for(int j=0;j<src->width;j++){
      if(i==0){
        y1=(double)((unsigned char)src->imageData[j]);
      }
      else{
        y1=(double)((unsigned char)src->imageData[(i-1)*src->widthStep+j]);
      }

      if(i==src->height-1){
        y2=(double)((unsigned char)src->imageData[i*src->widthStep+j]);
      }
      else{
        y2=(double)((unsigned char)src->imageData[(i+1)*src->widthStep+j]);
      }
      gradient_y = (y2-y1)/2;
      dst2->data.fl[i*dst2->cols+j]=gradient_y;
    }
  }

  double gradient_xy;
 
  for(int i=0;i<src->height;i++){
    for(int j=0;j<src->width;j++){
      gradient_xy=dst1->data.fl[i*dst1->cols+j]*dst2->data.fl[i*dst2->cols+j];
      dst3->data.fl[i*dst3->cols+j]=gradient_xy;
    }
  }

}


void check_local(CvMat *src,CvMat *dst){

  int local_max=0;
  int shift_r[] = {-1,-1,-1,0,0,1,1,1};
  int shift_c[] = {-1,0,1,-1,1,-1,0,1};

#pragma omp parallel for
  for(int i=1;i<src->rows-1;i++){
    for(int j=1;j<src->cols-1;j++){
      local_max=1;
      double local = src->data.fl[i*src->cols+j];
      for(int k=0;k<8;k++){
        if(src->data.fl[(i+shift_r[k])*src->cols+(j+shift_c[k])] > local){
          local_max=0;
          break;
        }
      }
      dst->data.fl[i*dst->cols+j]=(float)local_max;
    }
  }

}


void NMS(CvMat *src,CvMat *dst,int num,int R){

  double max=0;
  int mc=0,mr=0;
  CvMat *temp=0,*temp2=0;
  temp=cvCreateMat(src->rows,src->cols,CV_32FC1);
  temp2=cvCreateMat(src->rows,src->cols,CV_32FC1);

  cvCopy(src,temp,NULL);
 
  for(int i=0;i< temp->rows;i++){
    for(int j=0;j< temp->cols;j++){
      temp2->data.fl[i*temp2->cols+j] = 0;
    }
  }

 
  for(int kk=0;kk<500;kk++){
    max=0;
    for(int i=0;i< temp->rows;i++){
      for(int j=0;j< temp->cols;j++){
        if(temp->data.fl[i*src->cols+j]>max){
          max=temp->data.fl[i*src->cols+j];
          mc=j;
          mr=i;

        }
      }
    }
    //printf("kk:%d %d %d",kk,mr,mc);
    temp->data.fl[mr*temp->cols+mc]=0;
    temp2->data.fl[mr*temp->cols+mc]=1;
 
    for(int i=0;i<temp->rows;i++){
      for(int j=0;j<temp->cols;j++){
        if(temp->data.fl[i*temp->cols+j]>10){
          double dis=0;
          dis=(i-mr)*(i-mr)+(j-mc)*(j-mc);
          dis=pow(dis,0.5);
          if(dis<R){
            temp->data.fl[i*temp->cols+j]=0;
          }
        }
      }
    }


  }



  cvCopy(temp2,dst,NULL);

}
vector<double> getDiscriptor(IplImage *src,int x,int y,double angle){


  double d=0;
  vector<double> temp;
  temp.reserve(64); 
   
  for(int i=0;i<8;i++){
    for(int j=0;j<8;j++){
      for(int k=0;k<5;k++){
        d=0;
        for(int l=0;l<5;l++){
          int xp=0,yp=0;
          xp = cos(angle)*((i-4)*5+l) - sin(angle)*((j-4)*5+k) + x;
          yp = sin(angle)*((i-4)*5+l) + cos(angle)*((j-4)*5+k) + y;
          if(xp<0){
            xp=0;
          }
          else if(xp>=src->width){
            xp=src->width-1;
          }

          if(yp<0){
            yp=0;
          }
          else if(yp>=src->height){
            yp=src->height-1;
          }


          d += (unsigned char)src->imageData[(yp)*src->widthStep+xp];
        }
      }
      d=d/25.0;
      temp.push_back(d);
    }
  }

  double mean=0,var=0;
   
  for(int i=0;i<temp.size();i++){
    mean += temp[i];
    var += temp[i]*temp[i];
  }
  mean=mean/temp.size();
  var=var/temp.size();
  var=var-mean*mean;

  if(var==0){
    var=0.0001;
  }
   
  for(int i=0;i<temp.size();i++){
    temp[i]=(temp[i]-mean)/var;
  }

  return temp;
}





feature saveDiscriptor(IplImage *src,CvMat *point,CvMat *xG,CvMat *yG){
  feature temp;
  int boundary = 0;
  boundary = (int)20*pow(2.0,0.5);
 
  for(int i=0;i<point->rows;i++){
    for(int j=0;j<point->cols;j++){
      if(point->data.fl[i*point->cols+j]>0){

        double ang=0;
        ang =atan2(yG->data.fl[i*yG->cols+j],xG->data.fl[i*xG->cols+j]);
        temp.featureData.push_back(saveOnePointDiscriptor(src,j,i,ang));

      }
    }
  }

  return temp;
}
featurePoint saveOnePointDiscriptor(IplImage *src,int x,int y,double angle){
  featurePoint tempPoint;

  tempPoint.position=cvPoint(x,y);
  tempPoint.discriptor=getDiscriptor(src,x,y,angle);

  return tempPoint;

}



vector<pair<int ,int >> featureMatching(feature left, feature right,IplImage *src,IplImage *dst){


  vector<pair<int ,int >> Point;
  vector<pair<int ,int >> Match;
  //IplImage *out;
  //out = connect_and_blend(dst,src,0,0);
  //out = connect_and_blend(src,dst,0,0);
  ANNpointArray *descriptors = new ANNpointArray[2];
  ANNkd_tree **tree = new ANNkd_tree *[2];

  //left
  descriptors[0] = annAllocPts(left.featureData.size(), 64);
 
  for(int n = 0; n < left.featureData.size(); n++){
    for(int d = 0; d < 64; d++)
      descriptors[0][n][d] = left.featureData[n].discriptor[d];
  }
  tree[0] = new ANNkd_tree(descriptors[0],left.featureData.size(), 64);

  //right
  descriptors[1] = annAllocPts(right.featureData.size(), 64);

  for(int n = 0; n < right.featureData.size(); n++){
    for(int d = 0; d < 64; d++)
      descriptors[1][n][d] = right.featureData[n].discriptor[d];
  }
  tree[1] = new ANNkd_tree(descriptors[1],right.featureData.size(), 64);


  ANNpoint fd = annAllocPt(64);
  ANNidxArray nnIdx = new ANNidx[2];
  ANNdistArray dists = new ANNdist[2];

  for(int n = 0; n < left.featureData.size(); n++) {  

    for(int d = 0; d < 64; d++)
      fd[d] = descriptors[0][n][d];

    tree[1]->annkSearch(fd, 2, nnIdx, dists, 0);


    if(dists[0] < 0.6 * dists[1]) {
      pair<int,int> m; 
      m.first = n;
      m.second = nnIdx[0];
      Match.push_back(m);

      //cvLine(out,left.featureData[n].position,cvPoint(right.featureData[m.second].position.x+src->width,right.featureData[m.second].position.y),cvScalar(0,255,0),1,8,0);
    }
  }



  //cvNamedWindow("Matching");
  //cvShowImage("Matching",out);
  //cvSaveImage("match.jpg",out);
  //cvWaitKey(0);

  return Match;

}	

void parallel_connect_and_blend(IplImage *left_src,
                                     IplImage *right_src,
                                     CvRect paint_area,
                                     CvRect overlap_area,
                                     int height,
                                     int width,
                                     unsigned char *dst_data,
                                     int width_step,
                                     int y=0){

  int overlap_begin_x = overlap_area.x-paint_area.x;
  for(int i=0;i<paint_area.height;i++){
    unsigned char *dst_row_pointer = dst_data + (i+paint_area.y)*width_step + paint_area.x*3;
    unsigned char *left_src_row_pointer = (unsigned char*)left_src->imageData+i*left_src->widthStep+(left_src->width-overlap_area.width)*3;
    unsigned char *right_src_row_pointer = (unsigned char*)right_src->imageData+i*right_src->widthStep;
    //overlap area
    for(int j=0;j<paint_area.width;j++){
      if ( j<overlap_area.width ) {
        double a=0;
        a= (double)(j+1)/(double)overlap_area.width;
        *(dst_row_pointer+j*3)=(int)((double)(*left_src_row_pointer++) * (1-a) +
          (double)(*right_src_row_pointer++) * a);
        *(dst_row_pointer+j*3+1)=(int)((double)(*left_src_row_pointer++) * (1-a) +
          (double)(*right_src_row_pointer++) * a);
        *(dst_row_pointer+j*3+2)=(int)((double)(*left_src_row_pointer++) * (1-a) +
          (double)(*right_src_row_pointer++) * a);
      }
      else {
        *(dst_row_pointer+j*3)=*right_src_row_pointer++;
        *(dst_row_pointer+j*3+1)=*right_src_row_pointer++;
        *(dst_row_pointer+j*3+2)=*right_src_row_pointer++;
      }
    }
  }
}
IplImage* cylindrical_projection(IplImage *src,double f,double s){

  IplImage *temp_dst;
  temp_dst=cvCreateImage(cvSize(src->width,src->height),IPL_DEPTH_8U,3);

  int sx0=0,sy0=0;
  sx0=(src->width+1)/2;
  sy0=(src->height+1)/2;

  int dx0=0,dy0=0;
  dx0=(temp_dst->width+1)/2;
  dy0=(temp_dst->height+1)/2;    


  //°O±oµù¸Ñ±¼ 
  //s=f;
  //

  double src_theta=0;
  src_theta=atan(src->width/2/f);

  double dst_theta;
  dst_theta = src->width;
  dst_theta=dst_theta/s/2;

  double theta_of_pixel=0;
  theta_of_pixel=(dst_theta*2)/(double)temp_dst->width;

  int start=temp_dst->width,end=0;

   
  for(int i=0;i<temp_dst->height;i++){
    double h=0;
    h= (double)i - (double)dy0;

    for(int j=0;j<temp_dst->width;j++){
      double xp=0,yp=0;
      double t=0;
      double x=0;


      t= (-1)*dst_theta + j * theta_of_pixel;
      if( t >=(-1)*src_theta && t <= src_theta){
        if(start > j){
          start =j;
        }
        if(end<j){
          end=j;
        }

        xp=tan(t)*f;
        yp=h*pow(xp*xp+f*f,0.5)/s;
        xp=xp+sx0;
        yp= yp + dy0;


        if(yp>=0 && yp< temp_dst->height){
          reconstruction(src,xp,yp,temp_dst,i,j);
        }
      }
    }
  }

  cvSetImageROI(temp_dst,cvRect(start,0,end-start+1,temp_dst->height));
  IplImage *dst=0;
  dst=cvCreateImage(cvSize(end-start+1,temp_dst->height),IPL_DEPTH_8U,3);
  cvCopy(temp_dst,dst,NULL);

  cvReleaseImage(&temp_dst);
  return dst;

}




//bilinear reconstruction
//parameter:
//    src:the source image to rereconstruction
//    (x,y):the position to reconstruction
//    dst:the destination image to store
//    (row,column):the position of the destination to save
void reconstruction(IplImage *src,double x, double y,IplImage *dst,int row,int column){

  double R=0,G=0,B=0;

  double a= x-floor(x);
  double b= y-floor(y);
  int j=(int)floor(x);
  int i=(int)floor(y);

  //compute (1-a)(1-b) f[i,j]
  B = (1-a)*(1-b)*(unsigned char)src->imageData[i*src->widthStep+j*3];
  G = (1-a)*(1-b)*(unsigned char)src->imageData[i*src->widthStep+j*3+1];
  R = (1-a)*(1-b)*(unsigned char)src->imageData[i*src->widthStep+j*3+2];

  if(x>=0 && y>=0){
    //compute a(1-b) f[i+1,j]
    if(i+1 < src->height){
      B+= a*(1-b)*(unsigned char)src->imageData[(i+1)*src->widthStep+j*3];
      G+= a*(1-b)*(unsigned char)src->imageData[(i+1)*src->widthStep+j*3+1];
      R+= a*(1-b)*(unsigned char)src->imageData[(i+1)*src->widthStep+j*3+2];

      //compute ab f[i+1,j+1]
      if(j+1 < src->width){
        B+= a*b*(unsigned char)src->imageData[(i+1)*src->widthStep+(j+1)*3];
        G+= a*b*(unsigned char)src->imageData[(i+1)*src->widthStep+(j+1)*3+1];
        R+= a*b*(unsigned char)src->imageData[(i+1)*src->widthStep+(j+1)*3+2];
      }
    }    

    //compute (1-a)b f[i,j+1]
    if(j+1 < src->width){
      B+= (1-a)*b*(unsigned char)src->imageData[i*src->widthStep+(j+1)*3];    
      G+= (1-a)*b*(unsigned char)src->imageData[i*src->widthStep+(j+1)*3+1];
      R+= (1-a)*b*(unsigned char)src->imageData[i*src->widthStep+(j+1)*3+2];
    }
  }

  //store the pixel
  dst->imageData[row*dst->widthStep+column*3]=(int)B;
  dst->imageData[row*dst->widthStep+column*3+1]=(int)G;
  dst->imageData[row*dst->widthStep+column*3+2]=(int)R;


}


vector<int> RANSAC(vector<CvPoint> left,vector<CvPoint> right,int width,int n,int k,double t){
  vector<int> bestT,tempT;   
  srand(time(NULL));
  int bestC=0;
   
  for(int i=0;i<k;i++){
    int inlier;

    //random sample
    vector<CvPoint> RANS_left,RANS_left_other;
    vector<CvPoint> RANS_right,RANS_right_other;

    RANS_left_other=left;
    RANS_right_other=right;


    if(left.size()>n){
      for(int i=0;i<n;i++){
        int index=0;
        index = rand() % RANS_left_other.size();
        RANS_left.push_back(RANS_left_other[index]);
        RANS_right.push_back(RANS_right_other[index]);
        RANS_left_other.erase(RANS_left_other.begin()+index,RANS_left_other.begin()+index+1);
        RANS_right_other.erase(RANS_right_other.begin()+index,RANS_right_other.begin()+index+1);
      }
    }
    else{
      RANS_left=left;
      RANS_right=right;
    }



    // get model
    tempT = Alignment(RANS_left,RANS_right,width);

    //compute every data with the miodel

    if(left.size()>n){
      int C=0;
      for(int i=0;i<RANS_left_other.size();i++){
        double distance=0,distance_x=0,distance_y=0;
        int xp=0,yp=0;
        xp =(RANS_right_other[i].x+width)-RANS_left_other[i].x-tempT[0];
        yp =RANS_right_other[i].y-RANS_left_other[i].y+tempT[1];
        distance=pow((double)(xp*xp+yp*yp),0.5);
        //distance_x=abs(xp);
        //distance_y=abs(yp);
        //printf("%d:%d %d\n",i,abs(xp),abs(yp));
        if(abs(xp) < t && abs(yp)<t ){

          C++;
        }
      }
      //printf("\nC:%d\n",C);
      if(C>=bestC && tempT[0]<width){
        bestC = C;
        bestT=tempT;
      }
    }

    else{
      bestT=tempT;
      break;
    }

  } 

  return bestT;
}


//input : 2 matched point vector and the left image's width
//output: a translation vector
//eg. toreturn[0]=the x direction's displacement
//    toreturn[1]=the number of pixel that right image should translae on y  
vector<int> Alignment(vector<CvPoint> left,vector<CvPoint> right,int width){
  vector<int> toreturn;

  int n=0;
  int x=0,y=0;
  n=left.size();
   
  for(int i=0;i<n;i++){
    x += ((right[i].x +width)-left[i].x);
    y += (left[i].y-right[i].y);
  }

  toreturn.push_back(x/n);
  toreturn.push_back(y/n);

  return toreturn;
}



#endif