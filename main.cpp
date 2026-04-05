#include <iostream>
#include <windows.h>
#include <vector>
#include "SpikeNet.h"
#include "consoleUI.h"

int main()
{
    snn::Network net;
    std::vector<double> inp(3, 0.0d);
    double time = 0.0d;
    std::vector<double> decay(3, 0.89d); //= {0.95d, 0.45d, 0.7d, 0.82d, 0.39d, 0.99d, 0.36d, 0.291d};
    double count = 60.0d;
    double signal = 5.0d;
    while(true) {
        if(time >= count) break;
        std::cout << "time = " << time << '\n';
        if (GetAsyncKeyState('A') & 0x8000) inp[0] = signal;
        if (GetAsyncKeyState('S') & 0x8000) inp[1] = signal/3;
        if (GetAsyncKeyState('D') & 0x8000) inp[2] = signal/4.5d;
        //if (GetAsyncKeyState('F') & 0x8000) inp[3] = signal;
        //if (GetAsyncKeyState('G') & 0x8000) inp[4] = signal;
        //if (GetAsyncKeyState('H') & 0x8000) inp[5] = signal;
        //if (GetAsyncKeyState('J') & 0x8000) inp[6] = signal;
        //if (GetAsyncKeyState('K') & 0x8000) inp[7] = signal;
        if (GetAsyncKeyState('Q') & 0x8000) break;

        for (int i = 0; i < 3; i++) {
            inp[i] *= decay[i];
        }

        net.step(time, inp, 0.0d);
        net.logSpikeVisual();
        
        time += dt;
        Sleep(1);
    }
    std::cout << '\n';
    net.printWeights();
    system("pause");
    return 0;
} 