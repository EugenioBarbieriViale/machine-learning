#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float train[][3] = {
	{0,0, 0},
	{1,0, 1},
	{0,1, 1},
	{1,1, 0},
};

#define train_count (int)sizeof(train) / (int)sizeof(train[0])
#define square(x) (x)*(x)

#define h 1e-3
#define N 10000
#define rate 20

float sigmoid(float x) {
	return 1.f/(1.f + expf(-x));
}

float d_sigmoid(float x) {
    return sigmoid(x)*(1-sigmoid(x));
}

float rand_float(void) {
	return (float) rand() / (float) RAND_MAX;
}

void init(float xor[]) {
    for (int i=0; i<train_count; i++) {
        xor[i] = rand_float();
    }
}

float forward(float xor[], float x1, float x2) {
	float y_or = sigmoid(xor[0] * x1 + xor[1] * x2 + xor[2]);
	float y_nand = sigmoid(xor[3] * x1 + xor[4] * x2 + xor[5]);
	return sigmoid(xor[6] * y_or + xor[7] * y_nand + xor[8]);
}

float loss(float xor[]) {
    float cost = 0;
    for (int i=0; i<train_count; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float label = train[i][2];

        cost += square(forward(xor, x1, x2) - label);
    }
    return (cost / train_count);
}

float gloss(float xor[], float g[]) {
    
}

int main() {
    float xor[9];
    float g[9];

    init(xor);
    float cost = loss(xor);
    printf("%f\n", cost);

    return 0;
}
