// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT runtime∕internal∕atomic·Casp(SB),NOSPLIT,$0
	B	runtime·cas(SB)

// This is only valid for ARMv6+, however, NaCl/ARM is only defined
// for ARMv7A anyway.
TEXT runtime∕internal∕atomic·Cas(SB),NOSPLIT,$0
	B	runtime∕internal∕atomic·armcas(SB)

TEXT runtime∕internal∕atomic·Casp1(SB),NOSPLIT,$0
	B	runtime∕internal∕atomic·Cas(SB)
