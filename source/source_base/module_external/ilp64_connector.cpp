#include <complex>
#include <mkl_types.h>

// The ILP64 BLAS connector provides the int-to-MKL_INT adapters as inline
// functions.  LibRI refers to these adapters from its own headers, so keep
// one out-of-line copy in the ABACUS base target for the final link.
#define inline
#include "ilp64_blas_connector.h"
#undef inline
