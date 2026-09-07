// asmcheck -gcflags=-+ -gcflags=-p=internal/runtime/sys

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

// Calls to this function are replaced by the compiler intrinsic.
//
//go:noinline
func GetCallerPC() uintptr { return 0 }

func callerPC() uintptr {
	// The continuation slot is 8 bytes on both ports, but wasm32 stores
	// the logical PC in its low word and the function handle in its high word.
	// wasm:`I64Load\s` -`I64Load32U`
	return GetCallerPC()
}
