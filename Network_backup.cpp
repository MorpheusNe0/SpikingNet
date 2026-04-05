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
        void updateDopamine(size_t layer) {
            if (layer >= paramet_n.size()) return;
            bool spike_detected = false;
            for(int i = 0; i < paramet_n[layer].quantityN; i++) {
                if(neurons[paramet_n[layer].begin_layer_n + i].getSpike()) { spike_detected = true; 
                break;}
            }
            if(spike_detected) dopamine = max_dop; else dopamine = min_dop;
        }
        void step(double time, const std::vector<double>& SenSignals, double GlobalSignal) {
            neuroSignal(0, SenSignals);
            for(size_t i = 0; i < paramet_n.size(); i++) {
                stepLayer(i, GlobalSignal);
            }
            updateSynapse(time);
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