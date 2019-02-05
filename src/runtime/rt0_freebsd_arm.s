// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_arm_freebsd(SB),NOSPLIT,$0
	B	_rt0_arm(SB)

TEXT _rt0_arm_freebsd_lib(SB),NOSPLIT,$0
	B	_rt0_arm_lib(SB)
