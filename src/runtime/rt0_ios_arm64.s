// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// internal linking executable entry point.
// ios/arm64 only supports external linking.
TEXT _rt0_arm64_ios(SB),NOSPLIT|NOFRAME,$0
	UNDEF

// library entry point.
TEXT _rt0_arm64_ios_lib(SB),NOSPLIT|NOFRAME,$0
	JMP	_rt0_arm64_darwin_lib(SB)
