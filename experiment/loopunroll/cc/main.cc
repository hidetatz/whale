#include <time.h>
#include <iostream>
#include <cstdlib>

bool shouldUnroll() {
    const char* unrollEnv = std::getenv("UNROLL");
    if (unrollEnv != nullptr) {
        std::string unrollValue(unrollEnv);
        return unrollValue == "1";
    }

    return false;
}

int main() {
    bool unroll = shouldUnroll();
    std::cout << "unroll enabled: " << unroll << std::endl;

    const int n = 1024;
    int i, j, k;
    float **a = new float*[n];
    float **b = new float*[n];
    float **c = new float*[n];

    for (i = 0; i < n; i++) {
        a[i] = new float[n];
        b[i] = new float[n];
        c[i] = new float[n];
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            a[i][j] = (float)(int)(rand()/4096);
            b[i][j] = (float)(int)(rand()/4096);
            c[i][j] = 0.0f;
        }
    }

    clock_t startTime = clock();

    if (unroll) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j+=8) {
                for (k = 0; k < n; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                    c[i][j+1] += a[i][k] * b[k][j+1];
                    c[i][j+2] += a[i][k] * b[k][j+2];
                    c[i][j+3] += a[i][k] * b[k][j+3];
                    c[i][j+4] += a[i][k] * b[k][j+4];
                    c[i][j+5] += a[i][k] * b[k][j+5];
                    c[i][j+6] += a[i][k] * b[k][j+6];
                    c[i][j+7] += a[i][k] * b[k][j+7];
                }
            }
        }
    } else {
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                for (k = 0; k < n; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }

    clock_t stopTime = clock();

    float etime = (float)(stopTime - startTime) / CLOCKS_PER_SEC;
    printf("elapsed time = %15.7f sec\n", etime);
}
