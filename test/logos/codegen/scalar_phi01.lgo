// asmcheck -O=3

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// phiopt converts the 0/1 phi into a Boolean conversion. Correlated folding
// must see the aligned constant pairs before that representation change.

func phi01Int8(flag bool) int8 {
	var x, y int8
	if flag {
		x, y = 0, 7
	} else {
		x, y = 1, 6
	}
	// amd64:"MOVL [$]7, AX" -"(ADD|LEA|CMOV|XOR)"
	// arm64:"MOVD [$]7, R0" -"(ADD|CSEL|CSET|EOR)"
	return x + y
}

func phi01Uint8(flag bool) uint8 {
	var x, y uint8
	if flag {
		x, y = 0, 7
	} else {
		x, y = 1, 6
	}
	// amd64:"MOVL [$]7, AX" -"(ADD|LEA|CMOV|XOR)"
	// arm64:"MOVD [$]7, R0" -"(ADD|CSEL|CSET|EOR)"
	return x + y
}

func phi01Int16(flag bool) int16 {
	var x, y int16
	if flag {
		x, y = 0, 7
	} else {
		x, y = 1, 6
	}
	// amd64:"MOVL [$]7, AX" -"(ADD|LEA|CMOV|XOR)"
	// arm64:"MOVD [$]7, R0" -"(ADD|CSEL|CSET|EOR)"
	return x + y
}

func phi01Uint16(flag bool) uint16 {
	var x, y uint16
	if flag {
		x, y = 0, 7
	} else {
		x, y = 1, 6
	}
	// amd64:"MOVL [$]7, AX" -"(ADD|LEA|CMOV|XOR)"
	// arm64:"MOVD [$]7, R0" -"(ADD|CSEL|CSET|EOR)"
	return x + y
}

func phi01Int32(flag bool) int32 {
	var x, y int32
	if flag {
		x, y = 0, 7
	} else {
		x, y = 1, 6
	}
	// amd64:"MOVL [$]7, AX" -"(ADD|LEA|CMOV|XOR)"
	// arm64:"MOVD [$]7, R0" -"(ADD|CSEL|CSET|EOR)"
	return x + y
}

func phi01Uint32(flag bool) uint32 {
	var x, y uint32
	if flag {
		x, y = 0, 7
	} else {
		x, y = 1, 6
	}
	// amd64:"MOVL [$]7, AX" -"(ADD|LEA|CMOV|XOR)"
	// arm64:"MOVD [$]7, R0" -"(ADD|CSEL|CSET|EOR)"
	return x + y
}

func phi01Int64(flag bool) int64 {
	var x, y int64
	if flag {
		x, y = 0, 7
	} else {
		x, y = 1, 6
	}
	// amd64:"MOVL [$]7, AX" -"(ADD|LEA|CMOV|XOR)"
	// arm64:"MOVD [$]7, R0" -"(ADD|CSEL|CSET|EOR)"
	return x + y
}

func phi01Uint64(flag bool) uint64 {
	var x, y uint64
	if flag {
		x, y = 0, 7
	} else {
		x, y = 1, 6
	}
	// amd64:"MOVL [$]7, AX" -"(ADD|LEA|CMOV|XOR)"
	// arm64:"MOVD [$]7, R0" -"(ADD|CSEL|CSET|EOR)"
	return x + y
}

func phi01Reverse(flag bool) uint64 {
	var x, y uint64
	if flag {
		x, y = 1, 6
	} else {
		x, y = 0, 7
	}
	// amd64:"MOVL [$]7, AX" -"(ADD|LEA|CMOV|XOR)"
	// arm64:"MOVD [$]7, R0" -"(ADD|CSEL|CSET|EOR)"
	return x + y
}

func phi01Independent(left, right bool) uint64 {
	var x, y uint64
	if left {
		x = 0
	} else {
		x = 1
	}
	if right {
		y = 7
	} else {
		y = 6
	}
	// amd64:"LEAQ|ADDQ"
	// arm64:"ADD"
	return x + y
}
