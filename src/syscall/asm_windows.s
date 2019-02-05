// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// abi0Syms is a dummy symbol that creates ABI0 wrappers for Go
// functions called from assembly in other packages.
TEXT abi0Syms<>(SB),NOSPLIT,$0-0
	CALL ·getprocaddress(SB)
	CALL ·loadlibrary(SB)
