// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT errors(SB),$0
	V128Load8Lane	$0, $15
	V128Load8Lane	$0, $16			// ERROR "bad lane index"
	V128Load16Lane	$0, $7
	V128Load16Lane	$0, $8			// ERROR "bad lane index"
	V128Load32Lane	$0, $3
	V128Load32Lane	$0, $4			// ERROR "bad lane index"
	V128Load64Lane	$0, $1
	V128Load64Lane	$0, $2			// ERROR "bad lane index"
	V128Store8Lane	$0, $15
	V128Store8Lane	$0, $16			// ERROR "bad lane index"
	V128Store16Lane	$0, $7
	V128Store16Lane	$0, $8			// ERROR "bad lane index"
	V128Store32Lane	$0, $3
	V128Store32Lane	$0, $4			// ERROR "bad lane index"
	V128Store64Lane	$0, $1
	V128Store64Lane	$0, $2			// ERROR "bad lane index"
	V128Load64Lane	$8			// ERROR "bad lane index"
	V128Load64Lane	$-1, $0			// ERROR "bad offset"
	V128Load64Lane	$0x100000000, $0	// ERROR "bad offset"
	V128Load64Lane	x+0(FP), $0		// ERROR "bad offset"
	RET
