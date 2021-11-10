#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <sstream>
#include <array>
#include <memory>

namespace py = pybind11;

namespace {

void check_dim(const py::buffer_info &info, const std::string &name) {
    if (info.ndim != 2)
        throw std::runtime_error(name + " should have 2 dim");
}

template<typename TPixel=std::uint8_t, std::size_t BLOCK_SIZE=16>
class me_method {
  public:
    me_method(py::buffer ref_frame): ref_frame(ref_frame) {
        auto ref_info = ref_frame.request();
        check_frame(ref_info, "ref_frame");

        ref_data = static_cast<TPixel *>(ref_info.ptr);
        _ref_shape[0] = ref_info.shape[0];
        _ref_shape[1] = ref_info.shape[1];
        ref_linesize = ref_info.strides[0];
    }

    static constexpr auto block_size = BLOCK_SIZE;

  protected:
    static void check_frame(const py::buffer_info &info, const std::string &name) {
        check_dim(info, name);
        auto pixel_format = py::format_descriptor<TPixel>::format();
        if (info.format != pixel_format) {
            std::stringstream ss;
            ss << name << " should have format " << pixel_format;
            throw std::runtime_error(ss.str());
        }
        if (info.strides[1] != 1) {
            std::stringstream ss;
            ss << name << " should have stride[1] == 1, but got " << info.strides[1];
            throw std::runtime_error(ss.str());
        }
    }

    std::array<std::size_t, 2> num_blocks(const py::buffer_info &info) {
        return {
            (info.shape[0] - blocking_offset[0]) / BLOCK_SIZE,
            (info.shape[1] - blocking_offset[1]) / BLOCK_SIZE,
        };
    }

    void check_current_frame(const py::buffer_info &cur_info, const py::buffer_info &mv_info) {
        check_frame(cur_info, "cur_frame");
        auto nr_blocks = this->num_blocks(cur_info);
        if (static_cast<std::size_t>(mv_info.shape[0]) != nr_blocks[0] ||
                static_cast<std::size_t>(mv_info.shape[1]) != nr_blocks[1] ||
                mv_info.shape[2] != 2) {
            std::stringstream ss;
            ss << "expected mv shape (" << nr_blocks[0] << ", " << nr_blocks[1] << ", 2), "
                  "but got (" << mv_info.shape[0] << ", " << mv_info.shape[1] << ", " << mv_info.shape[2] << ")";
            throw std::runtime_error(ss.str());
        }
        auto mv_format = py::format_descriptor<int>::format();
        if (mv_info.format != mv_format) {
            throw std::runtime_error("mv should have format " + mv_format);
        }
        if (mv_info.strides[2] != 1) {
            std::stringstream ss;
            ss << "mv should have stride[2] == 1, but got " << mv_info.strides[1];
            throw std::runtime_error(ss.str());
        }
    }

    template<typename TBlock>
    void for_each_block(const py::buffer_info &cur_info, const py::buffer_info &mv_info, TBlock block) {
        int *p_mv_0 = static_cast<int *>(mv_info.ptr);
        for (std::size_t i = blocking_offset[0]; i <= cur_info.shape[0] - BLOCK_SIZE; i += BLOCK_SIZE) {
            p_mv_0 += mv_info.strides[0];

            auto p_mv_1 = p_mv_0;
            for (std::size_t j = blocking_offset[1]; j <= cur_info.shape[1] - BLOCK_SIZE; j += BLOCK_SIZE) {
                p_mv_1 += mv_info.strides[1];

                block(i, j, p_mv_1);
            }
        }
    }

    std::uint64_t cmp_sad(TPixel *p_ref, TPixel *p_cur, std::size_t cur_linesize) {
        uint64_t sad = 0;
        for (std::size_t i = 0; i < BLOCK_SIZE; i++)
            for (std::size_t j = 0; j < BLOCK_SIZE; j++)
                sad += std::abs(p_ref[i * ref_linesize + j] - p_cur[i * cur_linesize + j]);
        return sad;
    }

    TPixel *p_ref(std::size_t x, std::size_t y) {
        return ref_data + x * ref_linesize + y;
    }

  private:
    py::buffer ref_frame;
    TPixel *ref_data;
    std::array<std::size_t, 2> _ref_shape;
    std::size_t ref_linesize;

  public:
    std::array<std::size_t, 2> blocking_offset = {0, 0};
    const std::array<std::size_t, 2> &ref_shape() { return _ref_shape; }
    std::array<std::size_t, 2> num_blocks(py::buffer f) {
        return this->num_blocks(f.request());
    }
};

template<typename TPixel=std::uint8_t, std::size_t BLOCK_SIZE=16>
class esa : public me_method<TPixel, BLOCK_SIZE> {
    using base = me_method<TPixel, BLOCK_SIZE>;

  public:
    esa(py::buffer ref_frame, std::size_t search_range): base(ref_frame), search_range(search_range) {
    }

    void estimate(py::buffer cur_frame, py::buffer mv) {
        auto cur_info = cur_frame.request();
        auto mv_info = mv.request(true);
        base::check_current_frame(cur_info, mv_info);

        std::array<std::size_t, 2> max_search = {
            this->ref_shape()[0] - BLOCK_SIZE + 1,
            this->ref_shape()[1] - BLOCK_SIZE + 1,
        };

        this->for_each_block(cur_info, mv_info, [&](std::size_t x, std::size_t y, int *p_mv) {
            TPixel *p_cur = static_cast<TPixel *>(cur_info.ptr) + x * cur_info.strides[0] + y;
            std::array<std::size_t, 2> b_min {
                std::max(x - this->search_range, static_cast<std::size_t>(0)),
                std::max(y - this->search_range, static_cast<std::size_t>(0)),
            };
            std::array<std::size_t, 2> b_max {
                std::min(x + this->search_range, max_search[0]),
                std::min(y + this->search_range, max_search[1]),
            };
            uint64_t cost_min = this->cmp_sad(this->p_ref(x, y), p_cur, cur_info.strides[0]);
            if (cost_min == 0) {
                p_mv[0] = x;
                p_mv[1] = y;
                return;
            }

            for (std::size_t i = b_min[0]; i < b_max[0]; i++) {
                for (std::size_t j = b_min[1]; j < b_max[1]; j++) {
                    auto cost = this->cmp_sad(this->p_ref(i, j), p_cur, cur_info.strides[0]);
                    if (cost < cost_min) {
                        cost_min = cost;
                        p_mv[0] = i;
                        p_mv[1] = j;
                    }
                }
            }
        });
    }
  private:
    std::size_t search_range;
};

}

PYBIND11_MODULE(_C, m) {
    auto py_esa = py::class_<esa<>>(m, "ESA")
        .def_readonly_static("block_size", &esa<>::block_size)
        .def(py::init<py::buffer, std::size_t>(), py::arg("ref_frame"), py::arg("search_range"))
        .def_readwrite("blocking_offset", &esa<>::blocking_offset)
        .def("num_blocks", static_cast<std::array<std::size_t, 2> (esa<>::*)(py::buffer)>(&esa<>::num_blocks))
        .def("estimate", &esa<>::estimate, py::arg("cur_frame"), py::arg("mv"));
}
