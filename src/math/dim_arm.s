// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·Min(SB),NOSPLIT,$0
	B ·min(SB)

TEXT ·Max(SB),NOSPLIT,$0
	B ·max(SB)
