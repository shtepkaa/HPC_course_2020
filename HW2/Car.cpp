#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>

#define RGB_COMPONENT_COLOR 255
#define nthreads 5

struct PPMPixel {
    int red;
    int green;
    int blue;
};

typedef struct{
    int x, y, all;
    PPMPixel * data;
} PPMImage;

void readPPM(const char *filename, PPMImage& img){
    std::ifstream file (filename);
    if (file){
        std::string s;
        int rgb_comp_color;
        file >> s;
        if (s!="P3") {std::cout<< "error in format"<<std::endl; exit(9);}
        file >> img.x >>img.y;
        file >>rgb_comp_color;
        img.all = img.x*img.y;
        std::cout << "x=" << img.x << " y=" << img.y << " all=" <<img.all;
        img.data = new PPMPixel[img.all];
        for (int i=0; i<img.all; i++){
            file >> img.data[i].red >>img.data[i].green >> img.data[i].blue;
        }

    }else{
        std::cout << "the file:" << filename << "was not found" << std::endl;
    }
    file.close();
}

void writePPM(const char *filename, PPMImage & img){
    std::ofstream file (filename, std::ofstream::out);
    file << "P3"<<std::endl;
    file << img.x << " " << img.y << " "<< std::endl;
    file << RGB_COMPONENT_COLOR << std::endl;

    for(int i=0; i<img.all; i++){
        file << img.data[i].red << " " << img.data[i].green << " " << img.data[i].blue << (((i+1)%img.x ==0)? "\n" : " ");
    }
    file.close();
}

void changeColorPPM(PPMImage &img){
    for (int i=0; i<img.all; i++){
        img.data[i].red /= 2; 
    }
}

void shiftPPM(PPMImage &img)
{
    int n = 0;

    for (int shift = 0; shift < 900; shift ++)
    {
        #pragma omp parallel for shared(img) private(row,column) 
        
        for(int row = 1; row < img.y; row++)
        {
            int next;
            int current;

            for(int column = img.x; column >= 0; column--)
            {
                current = column + row*img.x;
                next = column + row*img.x - 1;

                img.data[current].red = img.data[next].red;
                img.data[current].green = img.data[next].green;
                img.data[current].blue = img.data[next].blue;
            }
        }
        if (shift % 10 == 0)
        {
            char filename[sizeof "car.ppm"];
            sprintf(filename, "image/%d.ppm", n++);
            writePPM(filename, img);

        }
    }
}

int main(){

    PPMImage image;

    readPPM("car.ppm", image);
    
    shiftPPM(image);

    delete(image.data);

    return 0;
}

