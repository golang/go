// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !gccgo

package abi

// FuncPC* intrinsics.
//
// CAREFUL: In programs with plugins, FuncPC* can return different values
// for the same function (because there are actually multiple copies of
// the same function in the address space). To be safe, don't use the
// results of this function in any == expression. It is only safe to
// use the result as an address at which to start executing code.

// FuncPCABI0 returns the entry PC of the function f, which must be a
// direct reference of a function defined as ABI0. Otherwise it is a
// compile-time error.
//
// Implemented as a compile intrinsic.
func FuncPCABI0(f any) uintptr

// FuncPCABIInternal returns the entry PC of the function f. If f is a
// direct reference of a function, it must be defined as ABIInternal.
// Otherwise it is a compile-time error. If f is not a direct reference
// of a defined function, it assumes that f is a func value. Otherwise
// the behavior is undefined.
//
// Implemented as a compile intrinsic.
func FuncPCABIInternal(f any) uintptr
