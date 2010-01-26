// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

// TODO(gri) Implement these routines.
TEXT ·addVV_s(SB),7,$0
	B ·addVV_g(SB)

TEXT ·subVV_s(SB),7,$0
	B ·subVV_g(SB)

TEXT ·addVW_s(SB),7,$0
	B ·addVW_g(SB)

TEXT ·subVW_s(SB),7,$0
	B ·subVW_g(SB)

TEXT ·mulAddVWW_s(SB),7,$0
	B ·mulAddVWW_g(SB)

TEXT ·addMulVVW_s(SB),7,$0
	B ·addMulVVW_g(SB)

TEXT ·divWVW_s(SB),7,$0
	B ·divWVW_g(SB)

