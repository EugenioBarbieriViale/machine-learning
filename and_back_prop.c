#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float train[][3] = {
	{0,0, 0},
	{1,0, 0},
	{0,1, 0},
	{1,1, 1},
};

#define train_count (int)sizeof(train) / (int)sizeof(train[0])
#define square(x) (x)*(x)

#define epochs 1000
#define rate 10

float sigmoid(float x) {
	return 1.f/(1.f + expf(-x));
}

float d_sigmoid(float x) {
    return sigmoid(x)*(1-sigmoid(x));
}

float rand_float(void) {
	return (float) rand() / (float) RAND_MAX;
}

float forward(float w1, float w2, float b, float x1, float x2) {
    return sigmoid(w1 * x1 + w2 * x2 + b);
}

float loss(float w1, float w2, float b) {
    float cost = 0;
    for (int i=0; i<train_count; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];

        float label = train[i][2];
        float out = forward(w1, w2, b, x1, x2);

        cost += square(out - label);
    }
    return (cost / train_count);
}

void gloss(float w1, float w2, float b, float *dw1, float *dw2, float *db) {
    for (int i=0; i<train_count; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];

        float label = train[i][2];
        float a = forward(w1, w2, b, x1, x2);

        *dw1 += 2 * (a - label) * a * (1 - a) * x1;
        *dw2 += 2 * (a - label) * a * (1 - a) * x2;
        *db  += 2 * (a - label) * a * (1 - a);
    }

    *dw1 /= train_count;
    *dw2 /= train_count;
    *db  /= train_count;
}

void update(float *w1, float *w2, float *b, float dw1, float dw2, float db) {
    *w1 -= rate * dw1;
    *w2 -= rate * dw2;
    *b  -= rate *  db;
}

void print_net(float w1, float w2, float b) {
    for (int i=0; i<train_count; i++) {
        float out = forward(w1, w2, b, train[i][0], train[i][1]);
        printf("%.1f - %.1f: (%.1f) -> %f\n", train[i][0], train[i][1], train[i][2], out);
    }
}

int main() {
    float w1 = rand_float();
    float w2 = rand_float();
    float b  = rand_float();

    float dw1 = 0;
    float dw2 = 0;
    float db  = 0;

    for (int i=0; i<epochs; i++) {
        gloss(w1, w2, b, &dw1, &dw2, &db);
        update(&w1, &w2, &b, dw1, dw2, db);

        float cost = loss(w1, w2, b);
        printf("%f\n", cost);
    }

    printf("-------------------------\n");
    print_net(w1, w2, b);

    return 0;
}
