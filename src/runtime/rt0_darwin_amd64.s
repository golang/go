// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_amd64_darwin(SB),NOSPLIT,$-8
	JMP	_rt0_amd64(SB)

// When linking with -shared, this symbol is called when the shared library
// is loaded.
TEXT _rt0_amd64_darwin_lib(SB),NOSPLIT,$0
	JMP	_rt0_amd64_lib(SB)
