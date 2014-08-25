// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls for Windows are implemented in ../runtime/syscall_windows.goc
//

#include "textflag.h"

// func compileCallback(fn interface{}, cleanstack bool) uintptr
TEXT ·compileCallback(SB),NOSPLIT,$0
	JMP	runtime·compileCallback(SB)
