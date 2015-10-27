// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_arm64_android(SB),NOSPLIT,$-8
	MOVD	$_rt0_arm64_linux(SB), R4
	B	(R4)

TEXT _rt0_arm64_android_lib(SB),NOSPLIT,$-8
	MOVD	$_rt0_arm64_linux_lib(SB), R4
	B	(R4)
