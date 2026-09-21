// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·noframe(SB), NOSPLIT|NOFRAME, $0-0
	XORL	BP, BP
	RET
