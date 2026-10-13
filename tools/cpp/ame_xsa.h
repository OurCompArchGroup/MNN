// Shim: the AME kernel interface has a single definition in the CPU backend
// (source/backend/cpu/compute/ame_xsa.h). Kept as a forwarding header because
// MNNV2Basic.cpp includes it by bare name.
#ifndef AME_XSA_TOOLS_SHIM_H
#define AME_XSA_TOOLS_SHIM_H
#include "../../source/backend/cpu/compute/ame_xsa.h"
#endif
