#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int main(){
    unsigned char buffer[10];
    FILE *ptr;
    ptr = fopen("data_batch_1.bin","rb");  // r for read, b for binary
    fread(buffer,sizeof(buffer),1,ptr); // read 10 bytes to our buffer 
    for(int i = 0; i<10; i++)
    printf("%u ", buffer[i]); // prints a series of bytes 
}