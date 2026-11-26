/*
code developed by
    Rong-Qiang He (rqhe@ruc.edu.cn, RUC, China)
date 2012 - 2017
*/

#ifndef _BNMLFAST_H_
#define _BNMLFAST_H_

#include "stdpfx.h"
#include "basis_index.h"

class BnmlFast {
public:
    // The read-only Pascal table is shared by all NocSpace/Onb instances.
    Idx storage_elements() const { return 0; }
    Idx storage_dynamic_bytes() const { return 0; }
    BasisIndex operator()(Int n, Int m) const;
};

#endif /* _BNMLFAST_H_ */
