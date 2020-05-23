#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <time.h>

#include "genann.h"

/* Number of training samples to load */
#define SAMPLES 500
#define NUM_PIXELS 3072
#define NUM_CLASSES 10

double *input, *class;
const char *bin_1 = "data_batch_1.bin";
unsigned char label_buffer;
unsigned char training_buffer [NUM_PIXELS];

void loadData(){
    /* Load the CIFAR-10 data-set. */
    FILE *in = fopen("data_batch_1.bin", "rb");
    if (!in) {
        printf("Could not open file: %s\n", bin_1);
        exit(1);
    }

    printf("Loading %d data points from %s\n", SAMPLES, bin_1);

    /* Allocate memory for input and output data. */
    input = malloc(sizeof(double) * SAMPLES * NUM_PIXELS);
    class = malloc(sizeof(double) * SAMPLES * NUM_CLASSES);

    /* Read the file into our arrays. */
    int i, j;
    for (i = 0; i < SAMPLES; ++i) {
        double *p = input + i * NUM_PIXELS;
        double *c = class + i * NUM_CLASSES;

        // Initialize all class outputs to 0.
        for(j = 0; j < NUM_CLASSES; ++j){
            c[j] = 0;
        }

        // Read in image class and values.
        fread(&label_buffer,sizeof(label_buffer),1,in);  
        fread(&training_buffer,sizeof(training_buffer),1,in);

        // Set normalized pixel values
        for (j = 0; j < NUM_PIXELS; ++j){
            p[j] = training_buffer[j] / 255;
        }

        // Set correct label index as 1.
        c[label_buffer] = 1;
    }
    fclose(in);
}

int main(){
    printf("Training an ANN on the CIFAR dataset using backpropagation.\n");
    srand(time(0));

    /* Load the data from file. */
    loadData();

    /* 3072 inputs.
     * 1 hidden layer(s) of 10 neurons.
     * 10 outputs (1 per class)
     */
    genann *ann = genann_init(NUM_PIXELS, 2, 100, 10);
    int i, j;
    int loops = 100;

    /* Train the network with backpropagation. */
    printf("Training for %d loops over data.\n", loops);
    for (i = 0; i < loops; ++i) {
        for (j = 0; j < SAMPLES; ++j) {
            genann_train(ann, input+j*NUM_PIXELS, class+j*NUM_CLASSES, .01);
        }
    }

    // Evaluate model accuracy
    int correct = 0;
    for (j = 0; j < SAMPLES; ++j) {
        const double *guess = genann_run(ann, input+j*NUM_PIXELS);

        // Get max index which will be our best guess.
        int max_index = 0;
        double max = guess[max_index];
        for (i = 0; i < 10; ++i){
            if (guess[i] > max){
                max = guess[i];
                max_index = i;
            }
        }

        //printf("%d ", max_index);

        // Check actual with guess
        if(class[j*3+0] == 1.0){
            if (max_index == 0) {++correct;}
        } else if (class[j*3+1] == 1.0){
            if (max_index == 1) {++correct;}
        } else if (class[j*3+2] == 1.0){
            if (max_index == 2) {++correct;}
        } else if (class[j*3+3] == 1.0){
            if (max_index == 3) {++correct;}
        } else if (class[j*3+4] == 1.0){
            if (max_index == 4) {++correct;}
        } else if (class[j*3+5] == 1.0){
            if (max_index == 5) {++correct;}
        } else if (class[j*3+6] == 1.0){
            if (max_index == 6) {++correct;}
        } else if (class[j*3+7] == 1.0){
            if (max_index == 7) {++correct;}
        } else if (class[j*3+8] == 1.0){
            if (max_index == 8) {++correct;}
        } else if (class[j*3+9] == 1.0){
            if (max_index == 9) {++correct;}
        } 
    }

    printf("%d/%d correct (%0.1f%%).\n", correct, SAMPLES, (double)correct / SAMPLES * 100.0);


    genann_free(ann);
    free(input);
    free(class);

    return 0;
}