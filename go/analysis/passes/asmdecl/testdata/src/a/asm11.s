// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build riscv64

// writing to result in ABIInternal function
TEXT ·returnABIInternal<ABIInternal>(SB), NOSPLIT, $8
	MOV	$123, X10
	RET
TEXT ·returnmissingABIInternal<ABIInternal>(SB), NOSPLIT, $8
	MOV	$123, X20
	RET // want `RET without writing to result register`
