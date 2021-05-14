// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

func FuncPCTestFn()

var FuncPCTestFnAddr uintptr // address of FuncPCTestFn, directly retrieved from assembly

//go:noinline
func FuncPCTest() uintptr {
	return FuncPCABI0(FuncPCTestFn)
}
