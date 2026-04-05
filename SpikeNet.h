#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#define dt 0.001d

namespace snn
{	
    /*
    enum NeuroType {
        Sensory, Integrator,
        Inhibitory, Neuromodulator,
        PredictionError, Prediction,
        GlobalWorkspace, Motor, ordinary
    }; */
    class Neuron
    {
    private:
        double V_m = 0;
        double I_syn = 0;
        int refractory = 0;

        double V_rest;
        double V_th;
        double tau_m;
        double tau_syn;

        double R;
        int t_ref;

        double alpha_m;
        double alpha_syn;

        bool spike = false;
        int type = 0;
    public:
        Neuron(int typ) {
            type = typ;
            switch (typ)
            {
                case 0:
                    V_rest = 0.0d;
                    V_th = 1.0d;
                    tau_m = 0.050d;
                    tau_syn = 0.150d;
                    R = 1.0d;
                    t_ref = 2;
                    alpha_m = exp(-dt/tau_m);
                    alpha_syn = exp(-dt/tau_syn);
                    break;
                case 1: // Sensory neuron
                    V_rest = 0.0d;
                    V_th = 0.8d;
                    tau_m = 0.008d;
                    tau_syn = 0.003d;
                    R = 1.0d;
                    t_ref = 2;
                    alpha_m = exp(-dt/tau_m);
                    alpha_syn = exp(-dt/tau_syn);
                    break;
                case 2: // Integrator neuron
                    V_rest = 0.0d;
                    V_th = 1.0d;
                    tau_m = 0.250d;
                    tau_syn = 0.050d;
                    R = 1.0d;
                    t_ref = 5;
                    alpha_m = exp(-dt/tau_m);
                    alpha_syn = exp(-dt/tau_syn);
                    break;
                case 3: // Inhibitory neuron
                    V_rest = 0.0d;
                    V_th = 0.8d;
                    tau_m = 0.005d;
                    tau_syn = 0.002d;
                    R = 1.0d;
                    t_ref = 1;
                    alpha_m = exp(-dt/tau_m);
                    alpha_syn = exp(-dt/tau_syn);
                    break;
                case 4:
                    V_rest = 0.0d;
                    V_th = 0.5d;
                    tau_m = 0.100d;
                    tau_syn = 0.050d;
                    R = 1.0d;
                    t_ref = 10;
                    alpha_m = exp(-dt/tau_m);
                    alpha_syn = exp(-dt/tau_syn);
                    break;
            }
        }
        void step(double I_ext)
        {
            if (refractory > 0) {
                refractory--;
                spike = false;
                V_m = V_rest;
                return;
            }
            V_m = V_rest + (V_m - V_rest) * alpha_m + (I_ext + I_syn) * R * (1 - alpha_m);
            I_syn *= alpha_syn;
            if (V_m >= V_th)
            {
                V_m = V_rest;
                spike = true;
                refractory = t_ref;
            } else {
                spike = false;
            }
        }
        int getType() { return type; }
        bool getSpike() {
            return spike;
        }
        void add_synaptic_current(double current) {
            I_syn += current;
        }
    };
    class Synapse
    {
    private:
        Neuron* pre;
        Neuron* post;
        double weight;
        double A_plus = 0.01d;
        double A_minus = 0.012d;
        double t_plus = 0.016d;
        double t_minus = 0.017d;
        double last_pre_time = -1000.0d; 
        double last_post_time = -1000.0d;
        double min_weight = -1.2d;
        double max_weight = 1.2d;
        bool stdp = false;
    public:
        Synapse(Neuron* pre_n, Neuron* post_n, double w)
        {
            pre = pre_n;
            post = post_n;
            weight = w;
        }
        Synapse(Neuron* pre_n, Neuron* post_n, double w, bool s)
        {
            stdp = s;
            pre = pre_n;
            post = post_n;
            weight = w;
        }
        void update(double time, double dopamine)
        {
            if (stdp) {
                double delta = 0.0d;
                if(last_post_time > 0 && pre->getSpike()) delta = -A_minus * exp(-(time - last_post_time) / t_minus);
                if(last_pre_time > 0 && post->getSpike()) delta = A_plus * exp(-(time - last_pre_time) / t_plus);
                weight += dopamine * delta;
                if(pre->getSpike()) last_pre_time = time;
                if(post->getSpike()) last_post_time = time;

                if(weight < min_weight) weight = min_weight;
                else if(weight > max_weight) weight = max_weight;
            }
            if(pre->getSpike()) {
                post->add_synaptic_current(weight);
            }
        }
        double getWeight() { return weight; }
    };
    class Network
    {
    private:
        std::vector<Neuron> neurons;
        std::vector<Synapse> synapses;
        std::vector<Neuron> neuromodulators;
        struct NLayer {
            size_t begin_layer_n;
            int quantityN;
            int typ;
            NLayer(size_t begin, int quan, int t) 
                : begin_layer_n(begin), quantityN(quan), typ(t) {}
        };
        struct ModParam {
            size_t post_n_number;
            std::vector<size_t> pre_conn_n_layer;
            std::vector<int> pre_conn_n_number;
        }
        std::vector<ModParam> spec;
        double dopamine_base = 0.5d;
        double dopamine_decay = 0.995d;
        double modulator_contribution = 0.2d;
        double max_dop = 1.5d;
        double min_dop = 0.0d;
        double dopamine = 0.75;
        std::vector<NLayer> paramet_n;
    public:
        void addLayerNeuron(int quan, int type) {
            paramet_n.emplace_back(neurons.size(), quan, type);
            for(int i = 0; i < quan; i++) {
                neurons.emplace_back(type);
            }
        }
        void fullyConection(size_t pre_layer, size_t post_layer, double weight, bool plastic) {
            for(int p = 0; p < paramet_n[pre_layer].quantityN; p++) {
                for(int n = 0; n < paramet_n[post_layer].quantityN; n++) {
                    synapses.emplace_back(&neurons[paramet_n[pre_layer].begin_layer_n + p], &neurons[paramet_n[post_layer].begin_layer_n + n], weight, plastic);
                }
            }
        }
        size_t getQuantityLayer(size_t layer) { 
            if (layer >= paramet_n.size()) return 0;
            return (size_t)paramet_n[layer].quantityN; 
        }
        void neuroSignal(size_t layer, const std::vector<double>& signals) {
            if (layer >= paramet_n.size()) return;
            if(paramet_n[layer].quantityN == signals.size()) {
                for(int i = 0; i < paramet_n[layer].quantityN; i++) {
                    neurons[paramet_n[layer].begin_layer_n + i].add_synaptic_current(signals[i]);
                }
            }
        }
        void updateSynapse(double time) {
            for(int i = 0; i < synapses.size(); i++) {
                synapses[i].update(time, dopamine);
            }
        }
        void stepLayer(size_t layer, double I_ext) {
            if (layer >= paramet_n.size()) return;
            for(int i = 0; i < paramet_n[layer].quantityN; i++) {
                neurons[paramet_n[layer].begin_layer_n + i].step(I_ext);
            }
        }
        void logSpikes(double time) {
            for (size_t i = 0; i < neurons.size(); i++) {
                if (neurons[i].getSpike()) {
                    std::cout << "n" << i << " @ " << time << "\n";
                }
            }
        }
        void printWeights() {
            for (size_t i = 0; i < synapses.size(); i++) {
                std::cout << "Synapse " << i << " weight = " << synapses[i].getWeight() << std::endl;
            }
        }
        void addModulator() {
            neuromodulators.emplace_back(4);
            spec.emplace_back();
        }
        void addConnModulator(size_t layer_pre_n, int number_pre_n, size_t number_post_n, double weight) {
            synapses.emplace_back(&neurons[paramet_n[layer_pre_n].begin_layer_n + number_pre_n], &neuromodulators[number_post_n], weight);
            spec[number_post_n].
        }
        void updateDopamine() {
            dopamine = dopamine_base + (dopamine - dopamine_base) * dopamine_decay;
            double amount_dophamine = 0.0d;
            for(int i = 0; i < spec.size(); i++) {
                if(neuromodulators[i].getSpike())
                {
                    for(int j = 0; j < spec[i].pre_conn_n_layer.size(); j++) {
                        if(neurons[paramet_n[spec[i].pre_conn_n_layer[j]].begin_layer_n + spec[i].pre_conn_n_number[j]].getSpike()) {
                            amount_dophamine += modulator_contribution;
                        }
                    }
                }
            }
            dopamine += amount_dophamine;
        }
        void step(double time, const std::vector<double>& SenSignals, double GlobalSignal) {
            neuroSignal(0, SenSignals);
            for(size_t i = 0; i < paramet_n.size(); i++) {
                stepLayer(i, GlobalSignal);
            }
            updateSynapse(time);
            updateDopamine();
        }
        void logSpikeVisual() {
            std::string line;
            for (int i = 0; i < neurons.size(); i++) {
                line += neurons[i].getSpike() ? '@' : '.';
            }
            std::cout << line;
        }
        void lateralConection(size_t layer, double weight, bool plastic) {
            if (layer >= paramet_n.size()) return;
            for(int i = 0; i < paramet_n[layer].quantityN; i++) {
                for(int n = 0; n < paramet_n[layer].quantityN; n++) {
                    if (i != n)
                    {synapses.emplace_back(&neurons[paramet_n[layer].begin_layer_n + i], &neurons[paramet_n[layer].begin_layer_n + n], weight, plastic);}
                }
            }
        }
        void customConnection(size_t pre_layer, int number_preN, size_t post_layer, int number_postN, double weight, bool plastic) {
            if (number_preN >= paramet_n[pre_layer].quantityN) return;
            if (number_postN >= paramet_n[post_layer].quantityN) return;
            synapses.emplace_back(&neurons[paramet_n[pre_layer].begin_layer_n + number_preN],
            &neurons[paramet_n[post_layer].begin_layer_n + number_postN], weight, plastic);
        }
    };
}