#ifndef INCLUDE_MODELS_MODELCLS_H
#define INCLUDE_MODELS_MODELCLS_H

#include "NvInferPlugin.h"
#include <fstream>
#include <cmath>     
#include <algorithm> 

#include "common.h"
#include "common.inl"
#include "networkSpace.h"
#include "resizeNormalize.h"


class Classifier : public AlgorithmBase
{
public:
    explicit Classifier(
        const std::string& nvptrEngine_FilePath,
        std::shared_ptr<logger::CustomLogger>& logger
    ) : AlgorithmBase(nvptrEngine_FilePath, logger) {
    }

    ~Classifier() override {
        std::cout << "Classifier destructor called." << std::endl;
    }

    void postprocess();
    void draw_boxes(size_t save_img_max_num);
};


#endif  // INCLUDE_MODELS_MODELCLS_H