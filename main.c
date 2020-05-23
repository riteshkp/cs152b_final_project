#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#define TRAIN_SIZE 10
#define NUM_PIXELS 3072

int main(){

    // Variables to load training data
    unsigned char label_buffer;
    unsigned char training_buffer [NUM_PIXELS];
    int labels [TRAIN_SIZE];
    int training_data [TRAIN_SIZE] [NUM_PIXELS];

    // Read Binary File
    FILE *ptr;
    ptr = fopen("data_batch_1.bin","rb");

    // Load from binary file.
    for(int i=0;i<TRAIN_SIZE;i++){
        fread(&label_buffer,sizeof(label_buffer),1,ptr);  
        fread(&training_buffer,sizeof(training_buffer),1,ptr);
        labels[i] = label_buffer;
        for(int j=0;j<NUM_PIXELS;j++){
            training_data[i][j] = training_buffer[j]; 
        }
    }
}