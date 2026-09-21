// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// func ReadX15() uint64
TEXT ·ReadX15(SB), $0-8
	MOVQ	X15, ret+0(FP)
	RET
