#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <vector>

torch::Tensor conv_forward(
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor c_m,
    torch::Tensor c_h,
    torch::Tensor c_w,
    torch::Tensor c_m_occ,
    int64_t pad,
    int64_t rf,
    int64_t stride,
    int64_t d
);

torch::Tensor conv_backward_w(
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor c_m,
    torch::Tensor c_h,
    torch::Tensor c_w,
    torch::Tensor c_m_occ,
    int64_t pad,
    int64_t rf,
    int64_t stride,
    int64_t d,
    torch::Tensor grad_y
);

torch::Tensor conv_backward_x(
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor c_m,
    torch::Tensor c_h,
    torch::Tensor c_w,
    torch::Tensor c_m_occ,
    int64_t pad,
    int64_t rf,
    int64_t stride,
    int64_t d,
    torch::Tensor grad_y
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        [] (torch::Tensor x,
            torch::Tensor weights,
            torch::Tensor c_m,
            torch::Tensor c_h,
            torch::Tensor c_w,
            torch::Tensor c_m_occ,
            int64_t pad,
            int64_t rf,
            int64_t stride,
            int64_t d) {
            return conv_forward(x, weights, c_m, c_h, c_w, c_m_occ, pad, rf, stride, d);
        },
        "convolutional logic layer forward (CUDA)"
    );
    m.def(
        "backward_w",
        [] (torch::Tensor x,
            torch::Tensor weights,
            torch::Tensor c_m,
            torch::Tensor c_h,
            torch::Tensor c_w,
            torch::Tensor c_m_occ,
            int64_t pad,
            int64_t rf,
            int64_t stride,
            int64_t d,
            torch::Tensor grad_y) {
            return conv_backward_w(x, weights, c_m, c_h, c_w, c_m_occ, pad, rf, stride, d, grad_y);
        },
        "convolutional logic layer backward_w (CUDA)"
    );
    m.def(
        "backward_x",
        [] (torch::Tensor x,
            torch::Tensor weights,
            torch::Tensor c_m,
            torch::Tensor c_h,
            torch::Tensor c_w,
            torch::Tensor c_m_occ,
            int64_t pad,
            int64_t rf,
            int64_t stride,
            int64_t d,
            torch::Tensor grad_y) {
            return conv_backward_x(x, weights, c_m, c_h, c_w, c_m_occ, pad, rf, stride, d, grad_y);
        },
        "convolutional logic layer backward_x (CUDA)"
    );
}