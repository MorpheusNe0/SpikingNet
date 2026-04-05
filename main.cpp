#include <iostream>
#include <windows.h>
#include <vector>
#include "SpikeNet.h"
#include "consoleUI.h"

int main()
{
    snn::Network net;
    net.addLayerNeuron(3, 1);
    net.addLayerNeuron(2, 3);
    net.addLayerNeuron(3, 2);
    net.addLayerNeuron(2, 0);
    net.addLayerNeuron(1, 0);
    net.addModulator(0.7d);
    net.addConnModulator(4, 0, 0, 0.5d, false);
    net.randomConnection(0, 1, 0.6, 0.1, 0.8, 0.8); 
    net.randomConnection(0, 2, 0.7, 0.2, 1.0, 0.8);
    net.randomConnection(1, 2, 0.5, -0.6, -0.1, 0.7);
    net.randomConnection(2, 3, 0.8, 0.3, 0.9, 0.8);  
    net.randomConnection(3, 4, 1.0, 0.4, 0.7, 0.8);
    net.randomConnection(1, 3, 0.7d, -0.4d, 0.0d, 0.9d);
    std::vector<double> inp(3, 0.0d);
    double time = 0.0d;
    double decay = 0.80d;
    double count = 60.0d;
    double signal = 3.0d;
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
        for (auto& i : inp) {
            i *= decay;
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