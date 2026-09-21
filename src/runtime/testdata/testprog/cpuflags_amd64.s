// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT	·checkAVX(SB), NOSPLIT|NOFRAME, $0-0
	VXORPS	X1, X2, X3
	RET
