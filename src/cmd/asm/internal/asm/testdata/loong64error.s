// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT errors(SB),$0
	VSHUF4IV	$16, V1, V2	// ERROR "operand out of range 0 to 15"
	XVSHUF4IV	$16, X1, X2	// ERROR "operand out of range 0 to 15"
