#define PY_SSIZE_T_CLEAN
#include <Python.h>

static struct PyModuleDef pyme_ext = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_C",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__C(void) {
    PyObject *m = PyModule_Create(&pyme_ext);
    if (m == nullptr)
        return nullptr;

    return m;
}
