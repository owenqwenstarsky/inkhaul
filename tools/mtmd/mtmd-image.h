#pragma once

#include "ggml.h"
#include "mtmd-model.h"

#include <vector>
#include <string>

#define MTMD_INTERNAL_HEADER

struct mtmd_image_preproc_out {
    std::vector<mtmd_image_f32> entries;
    // grid size is required for llava-uhd style models

    mtmd_image_f32 overview; // overview image (downscaled image)
    int grid_x = 0;
    int grid_y = 0;

    void append(const mtmd_hparams & hparams, const mtmd_image_u8 & img, bool normalized = true);
    void append(const mtmd_hparams & hparams, const std::vector<mtmd_image_u8> & imgs, bool normalized = true);
    void append(const mtmd_hparams & hparams, mtmd_image_f32 & img, bool normalized = true);

    void append_overview(const mtmd_hparams & hparams, const mtmd_image_u8 & img, bool normalized = true);
    bool has_overview() const {
        return overview.nx() > 0 || overview.ny() > 0;
    }
};

// base class, models must inherit from this class
struct mtmd_image_preprocessor {
    const mtmd_hparams & hparams;

    mtmd_image_preprocessor(const mtmd_model * model): hparams(model->hparams) {}
    mtmd_image_preprocessor(const mtmd_hparams & hparams): hparams(hparams) {}

    virtual ~mtmd_image_preprocessor() = default;
    virtual mtmd_image_preproc_out preprocess(const mtmd_image_u8 & img) const = 0;
    // frames of a video, which the video helper may have already sized
    virtual mtmd_image_preproc_out preprocess_video_frame(const mtmd_image_u8 & img) const { return preprocess(img); }
};

// pad bottom-right to a multiple of patch_size*n_merge; resize only if the image is above image_max_pixels
struct mtmd_image_preprocessor_inkling2v : mtmd_image_preprocessor {
    using mtmd_image_preprocessor::mtmd_image_preprocessor;
    mtmd_image_preproc_out preprocess(const mtmd_image_u8 & img) const override;
};
