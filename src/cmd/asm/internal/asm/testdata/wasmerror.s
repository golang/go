// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT errors(SB),$0
	I8x16ExtractLaneS	$15
	I8x16ExtractLaneS	$16		// ERROR "bad lane index"
	I8x16ExtractLaneU	$15
	I8x16ExtractLaneU	$16		// ERROR "bad lane index"
	I8x16ReplaceLane	$15
	I8x16ReplaceLane	$16		// ERROR "bad lane index"
	I16x8ExtractLaneS	$7
	I16x8ExtractLaneS	$8		// ERROR "bad lane index"
	I16x8ExtractLaneU	$7
	I16x8ExtractLaneU	$8		// ERROR "bad lane index"
	I16x8ReplaceLane	$7
	I16x8ReplaceLane	$8		// ERROR "bad lane index"
	I32x4ExtractLane	$3
	I32x4ExtractLane	$4		// ERROR "bad lane index"
	I32x4ReplaceLane	$3
	I32x4ReplaceLane	$4		// ERROR "bad lane index"
	I64x2ExtractLane	$1
	I64x2ExtractLane	$2		// ERROR "bad lane index"
	I64x2ReplaceLane	$1
	I64x2ReplaceLane	$2		// ERROR "bad lane index"
	F32x4ExtractLane	$3
	F32x4ExtractLane	$4		// ERROR "bad lane index"
	F32x4ReplaceLane	$3
	F32x4ReplaceLane	$4		// ERROR "bad lane index"
	F64x2ExtractLane	$1
	F64x2ExtractLane	$2		// ERROR "bad lane index"
	F64x2ReplaceLane	$1
	F64x2ReplaceLane	$2		// ERROR "bad lane index"
	F64x2ExtractLane	$0, $1
	F64x2ExtractLane	$0, $2		// ERROR "bad lane index"
	F64x2ExtractLane	$-1		// ERROR "bad lane index"
	F64x2ExtractLane	R0		// ERROR "bad lane index"
	RET
