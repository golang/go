// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·Log10(SB),NOSPLIT,$0
	B ·log10(SB)

TEXT ·Log2(SB),NOSPLIT,$0
	B ·log2(SB)
