// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

// TODO(gri) Implement these routines.
TEXT big·addVV_s(SB),7,$0
	B big·addVV_g(SB)

TEXT big·subVV_s(SB),7,$0
	B big·subVV_g(SB)

TEXT big·addVW_s(SB),7,$0
	B big·addVW_g(SB)

TEXT big·subVW_s(SB),7,$0
	B big·subVW_g(SB)

TEXT big·mulAddVWW_s(SB),7,$0
	B big·mulAddVWW_g(SB)

TEXT big·addMulVVW_s(SB),7,$0
	B big·addMulVVW_g(SB)

TEXT big·divWVW_s(SB),7,$0
	B big·divWVW_g(SB)

