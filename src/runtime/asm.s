// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#ifndef GOARCH_amd64
TEXT ·sigpanic0(SB),NOSPLIT,$0-0
	JMP	·sigpanic<ABIInternal>(SB)
#endif
