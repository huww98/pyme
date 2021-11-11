#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <sstream>
#include <array>
#include <memory>

namespace py = pybind11;

namespace {

template<typename TPixel=std::uint8_t, std::size_t BLOCK_SIZE=16>
class me_method {
  public:
    me_method(py::buffer ref_frame): ref_frame(ref_frame) {
        auto ref_info = ref_frame.request();
        check_frame(ref_info, "ref_frame");

        ref_data = static_cast<TPixel *>(ref_info.ptr);
        _ref_shape[0] = ref_info.shape[0];
        _ref_shape[1] = ref_info.shape[1];
        ref_linesize = ref_info.strides[0] / sizeof(TPixel);
        assert(ref_linesize * sizeof(TPixel) == ref_info.strides[0]);
    }

    static constexpr auto block_size = BLOCK_SIZE;

  protected:
    static void check_frame(const py::buffer_info &info, const std::string &name) {
        if (info.ndim != 2)
            throw std::runtime_error(name + " should have 2 dim");
        auto pixel_format = py::format_descriptor<TPixel>::format();
        if (info.format != pixel_format) {
            std::stringstream ss;
            ss << name << " should have format " << pixel_format;
            throw std::runtime_error(ss.str());
        }
        py::ssize_t pix_stride = sizeof(TPixel);
        if (info.strides[1] != pix_stride) {
            std::stringstream ss;
            ss << name << " should have stride[1] == " << pix_stride << ", but got " << info.strides[1];
            throw std::runtime_error(ss.str());
        }
    }

    std::array<std::size_t, 2> num_blocks(const py::buffer_info &info) {
        return {
            info.shape[0] / BLOCK_SIZE,
            info.shape[1] / BLOCK_SIZE,
        };
    }

    void check_current_frame(const py::buffer_info &cur_info, const py::buffer_info &mv_info, const py::buffer_info &cost_info) {
        check_frame(cur_info, "cur_frame");
        if (mv_info.ndim != 3)
            throw std::runtime_error("mv should have 3 dim");
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
        py::ssize_t mv_stride = sizeof(int);
        if (mv_info.strides[2] != mv_stride) {
            std::stringstream ss;
            ss << "mv should have stride[2] == " << mv_stride << ", but got " << mv_info.strides[2];
            throw std::runtime_error(ss.str());
        }

        if (cost_info.ndim != 2)
            throw std::runtime_error("cost should have 2 dim");
        if (static_cast<std::size_t>(cost_info.shape[0]) != nr_blocks[0] ||
                static_cast<std::size_t>(cost_info.shape[1]) != nr_blocks[1]) {
            std::stringstream ss;
            ss << "expected cost shape (" << nr_blocks[0] << ", " << nr_blocks[1] << "), "
                  "but got (" << cost_info.shape[0] << ", " << cost_info.shape[1] << ")";
            throw std::runtime_error(ss.str());
        }
        auto cost_format = py::format_descriptor<uint64_t>::format();
        if (cost_info.format != cost_format) {
            throw std::runtime_error("cost should have format " + cost_format);
        }
    }

    template<typename TBlock>
    void for_each_block(const py::buffer_info &cur_info, const py::buffer_info &mv_info, const py::buffer_info &cost_info, TBlock block) {
        constexpr auto B = static_cast<std::ptrdiff_t>(BLOCK_SIZE);
        char *p_mv_0 = static_cast<char *>(mv_info.ptr);
        char *p_cost_0 = static_cast<char *>(cost_info.ptr);
        for (std::ptrdiff_t i = 0; i <= cur_info.shape[0] - B; i += B) {
            auto p_mv_1 = p_mv_0;
            auto p_cost_1 = p_cost_0;
            for (std::ptrdiff_t j = 0; j <= cur_info.shape[1] - B; j += B) {
                auto p_mv = reinterpret_cast<int *>(p_mv_1);
                auto p_cost = reinterpret_cast<uint64_t *>(p_cost_1);
                p_mv[0] = p_mv[1] = -1;
                *p_cost = std::numeric_limits<uint64_t>::max();
                block(i, j, p_mv, *p_cost);

                p_mv_1 += mv_info.strides[1];
                p_cost_1 += cost_info.strides[1];
            }
            p_mv_0 += mv_info.strides[0];
            p_cost_0 += cost_info.strides[0];
        }
    }

    std::uint64_t cmp_sad(TPixel *p_ref, TPixel *p_cur, std::size_t cur_linesize) {
        uint64_t sad = 0;
        for (std::size_t i = 0; i < BLOCK_SIZE; i++)
            for (std::size_t j = 0; j < BLOCK_SIZE; j++)
                sad += std::abs(p_ref[i * ref_linesize + j] - p_cur[i * cur_linesize + j]);
        return sad;
    }

    bool p_ref_vaild(std::ptrdiff_t x, std::ptrdiff_t y) {
        return x >= 0 && y >= 0 && x < ref_shape()[0] && y < ref_shape()[1];
    }

    std::array<std::ptrdiff_t, 2> ref_max_cmp() {
        constexpr auto B = static_cast<std::ptrdiff_t>(BLOCK_SIZE);
        return {
            this->ref_shape()[0] - B + 1,
            this->ref_shape()[1] - B + 1,
        };
    }

    bool p_ref_vaild_cmp(std::ptrdiff_t x, std::ptrdiff_t y) {
        auto m = ref_max_cmp();
        return x >= 0 && y >= 0 && x < m[0] && y < m[1];
    }

    TPixel *p_ref(std::size_t x, std::size_t y) {
        return ref_data + x * ref_linesize + y;
    }

  private:
    py::buffer ref_frame;
    TPixel *ref_data;
    std::array<std::ptrdiff_t, 2> _ref_shape;
    std::ptrdiff_t ref_linesize;

  public:
    const std::array<std::ptrdiff_t, 2> &ref_shape() { return _ref_shape; }
    std::array<std::size_t, 2> num_blocks(py::buffer f) {
        return this->num_blocks(f.request());
    }
};

template<typename TPixel=std::uint8_t, std::size_t BLOCK_SIZE=16>
class esa : public me_method<TPixel, BLOCK_SIZE> {
    using base = me_method<TPixel, BLOCK_SIZE>;

  public:
    esa(py::buffer ref_frame, std::size_t search_range, std::array<std::ptrdiff_t, 2> ref_offset)
        : base(ref_frame), search_range(search_range), _ref_offset(ref_offset) {
    }

    void estimate(py::buffer cur_frame, py::buffer mv, py::buffer cost) {
        auto cur_info = cur_frame.request();
        auto mv_info = mv.request(true);
        auto cost_info = cost.request(true);
        base::check_current_frame(cur_info, mv_info, cost_info);

        this->for_each_block(cur_info, mv_info, cost_info, [&](std::size_t x, std::size_t y, int *p_mv, uint64_t &cost) {
            auto cur_linesize = cur_info.strides[0] / sizeof(TPixel);
            TPixel *p_cur = static_cast<TPixel *>(cur_info.ptr) + x * cur_linesize + y;
            std::ptrdiff_t x_ref = x + this->_ref_offset[0];
            std::ptrdiff_t y_ref = y + this->_ref_offset[1];
            auto r = static_cast<std::ptrdiff_t>(this->search_range);
            std::array<std::ptrdiff_t, 2> b_min {
                std::max(x_ref - r, static_cast<std::ptrdiff_t>(0)),
                std::max(y_ref - r, static_cast<std::ptrdiff_t>(0)),
            };
            auto max_search = this->ref_max_cmp();
            std::array<std::ptrdiff_t, 2> b_max {
                std::min(x_ref + r + 1, max_search[0]),
                std::min(y_ref + r + 1, max_search[1]),
            };
            if (this->p_ref_vaild_cmp(x_ref, y_ref)) {
                cost = this->cmp_sad(this->p_ref(x_ref, y_ref), p_cur, cur_linesize);
                if (cost == 0) {
                    p_mv[0] = x_ref;
                    p_mv[1] = y_ref;
                    return;
                }
            }

            for (auto i = b_min[0]; i < b_max[0]; i++) {
                for (auto j = b_min[1]; j < b_max[1]; j++) {
                    auto c = this->cmp_sad(this->p_ref(i, j), p_cur, cur_linesize);
                    if (c < cost) {
                        cost = c;
                        p_mv[0] = i;
                        p_mv[1] = j;
                    }
                }
            }
        });
    }
  private:
    std::size_t search_range;
    std::array<std::ptrdiff_t, 2> _ref_offset;

  public:
    std::array<std::ptrdiff_t, 2> ref_offset() {
        return _ref_offset;
    }
};

}

PYBIND11_MODULE(_C, m) {
    auto py_esa = py::class_<esa<>>(m, "ESA")
        .def_readonly_static("block_size", &esa<>::block_size)
        .def_property_readonly("ref_offset", &esa<>::ref_offset)
        .def(py::init<py::buffer, std::size_t, std::array<std::ptrdiff_t, 2>>(),
            py::arg("ref_frame"), py::arg("search_range"), py::arg("ref_offset")=std::array<std::ptrdiff_t, 2>{0,0})
        .def("num_blocks", static_cast<std::array<std::size_t, 2> (esa<>::*)(py::buffer)>(&esa<>::num_blocks), py::arg("frame"))
        .def("estimate", &esa<>::estimate, py::arg("cur_frame"), py::arg("mv"), py::arg("cost"));
}
