//³¯¶h²» R00922011
#include<stdio.h>
#include<stdlib.h>
#include<highgui.h>
#include<cv.h>
#include<algorithm>
#include<math.h>
#include<vector>
#include<string>
#include "getFeaturePoint.h"

using namespace std;



feature getFeature(string name){
    
	
	/*
    printf("%d %d %d\n",img1.point.size(),img1.point[0].x,img1.point[0].y);
    printf("%d\n",img1.dis.size());
    for(int i=0;i<64;i++){
        printf("%d ",img1.dis[0].data[i]);
    }
    printf("\n");
    printf("\n");
    
   
    printf("%d %d %d\n",img2.point.size(),img2.point[0].x,img2.point[0].y);
    printf("%d\n",img2.dis.size());
    for(int i=0;i<64;i++){
        printf("%d ",img2.dis[0].data[0]);
    }
        printf("\n");
	*/
	
	
	
	feature temp;
    FILE *in = fopen(name.c_str(),"r");
    int x=0,y=0;
    double a=0;
    int pix=0;
    
    
    while(fscanf(in,"%d %d %lf",&x,&y,&a)==3){
        temp.point.push_back(cvPoint(x,y));
        discriptor d;
        for(int i=0;i<64;i++){
            fscanf(in,"%d",&pix);
            d.data.push_back(pix);
        }
        temp.dis.push_back(d);
    }
    
    
    return temp;
}
    
