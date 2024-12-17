#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double train[][3] = {
	{0,0, 0},
	{1,0, 0},
	{0,1, 0},
	{1,1, 1},
};

#define train_count (int)sizeof(train) / (int)sizeof(train[0])
#define square(x) (x)*(x)

#define epochs 1000
#define rate 10

double sigmoid(double x) {
	return 1.f/(1.f + expf(-x));
}

double d_sigmoid(double x) {
    return sigmoid(x)*(1-sigmoid(x));
}

double rand_double(void) {
	return (double) rand() / (double) RAND_MAX;
}

double forward(double w1, double w2, double b, double x1, double x2) {
    return sigmoid(w1 * x1 + w2 * x2 + b);
}

double loss(double w1, double w2, double b) {
    double cost = 0;
    for (int i=0; i<train_count; i++) {
        double x1 = train[i][0];
        double x2 = train[i][1];

        double label = train[i][2];
        double out = forward(w1, w2, b, x1, x2);

        cost += square(out - label);
    }
    return (cost / train_count);
}

void gloss(double w1, double w2, double b, double *dw1, double *dw2, double *db) {
    for (int i=0; i<train_count; i++) {
        double x1 = train[i][0];
        double x2 = train[i][1];

        double label = train[i][2];
        double a = forward(w1, w2, b, x1, x2);

        *dw1 += 2 * (a - label) * a * (1 - a) * x1;
        *dw2 += 2 * (a - label) * a * (1 - a) * x2;
        *db  += 2 * (a - label) * a * (1 - a);
    }

    *dw1 /= train_count;
    *dw2 /= train_count;
    *db  /= train_count;
}

void update(double *w1, double *w2, double *b, double dw1, double dw2, double db) {
    *w1 -= rate * dw1;
    *w2 -= rate * dw2;
    *b  -= rate *  db;
}

void print_net(double w1, double w2, double b) {
    for (int i=0; i<train_count; i++) {
        double out = forward(w1, w2, b, train[i][0], train[i][1]);
        printf("%.1f - %.1f: (%.1f) -> %f\n", train[i][0], train[i][1], train[i][2], out);
    }
}

int main() {
    double w1 = rand_double();
    double w2 = rand_double();
    double b  = rand_double();

    double dw1 = 0;
    double dw2 = 0;
    double db  = 0;

    for (int i=0; i<epochs; i++) {
        gloss(w1, w2, b, &dw1, &dw2, &db);
        update(&w1, &w2, &b, dw1, dw2, db);

        double cost = loss(w1, w2, b);
        printf("%f\n", cost);
    }

    printf("-------------------------\n");
    print_net(w1, w2, b);

    return 0;
}
