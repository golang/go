// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !wasm32

package runtime

import (
	"internal/abi"
	"internal/runtime/sys"
)

// Function value code words are entry PCs on these architectures.
//
//go:nosplit
func funcValuePC(fn uintptr) uintptr { return fn }

//go:nosplit
func funcHandle(pc uintptr) uintptr { return pc }

//go:nosplit
func gobufSetPC(buf *gobuf, pc uintptr) {
	buf.pc = pc
	buf.lr = 0
}

// gobufSetGoexit leaves PC-1 within goexit for stack unwinding.
//
//go:nosplit
func gobufSetGoexit(buf *gobuf) {
	gobufSetPC(buf, abi.FuncPCABI0(goexit)+sys.PCQuantum)
}
