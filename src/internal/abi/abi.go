// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

// RegArgs is a struct that has space for each argument
// and return value register on the current architecture.
type RegArgs struct {
	Ints   [IntArgRegs]uintptr
	Floats [FloatArgRegs]uint64
}
