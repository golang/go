// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_amd64_illumos(SB),NOSPLIT,$-8
	JMP	_rt0_amd64(SB)

TEXT _rt0_amd64_illumos_lib(SB),NOSPLIT,$0
	JMP	_rt0_amd64_lib(SB)
