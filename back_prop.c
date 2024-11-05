#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train[][2] = {
	{0,0},
	{1,2},
	{2,4},
	{3,6},
	{4,8},
};

#define train_count (int)sizeof(train) / (int)sizeof(train[0])
#define square(x) (x)*(x)

float rand_float(void) {
	return (float) rand() / (float) RAND_MAX;
}

float forward(float input, float w) {
    return input * w;
}

float loss(float input, float label, float w) {
    return square(label - forward(input, w));
}

int main() {
    srand(time(NULL));
    float w = rand_float();

    const int epochs = 20;
    const float h = 1e-2;
    const float rate = 1e-1;

    float cost = 0;

    for (int i=0; i<epochs; i++) {
        for (int j=0; j<train_count; j++) {
            float input = train[j][0];
            float label = train[j][1];

            cost = loss(input, label, w);

#if 0
            float dw = (loss(input, label, w + h) - loss(input, label, w)) / h;
#else
            float dw = 2 * (forward(input, w) - label) * input;
#endif

            w -= dw * rate;
        }
        printf("cost: %f  weight: %f\n", cost, w);
    }
}
