// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT	·socketcall(SB),NOSPLIT,$0-36
	JMP	syscall·socketcall(SB)

TEXT	·rawsocketcall(SB),NOSPLIT,$0-36
	JMP	syscall·socketcall(SB)
