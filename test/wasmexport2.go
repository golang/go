// errorcheck

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that wasmexport supports allowed types and rejects
// unallowed types.

//go:build wasm

package p

import (
	"structs"
	"unsafe"
)

//go:wasmexport good1
func good1(int32, uint32, int64, uint64, float32, float64, unsafe.Pointer) {} // allowed types

type MyInt32 int32

//go:wasmexport good2
func good2(MyInt32) {} // named type is ok

//go:wasmexport good3
func good3() int32 { return 0 } // one result is ok

//go:wasmexport good4
func good4() unsafe.Pointer { return nil } // one result is ok

//go:wasmexport good5
func good5(string, uintptr) bool { return false } // bool, string, and uintptr are allowed

//go:wasmexport bad1
func bad1(any) {} // ERROR "go:wasmexport: unsupported parameter type"

//go:wasmexport bad2
func bad2(func()) {} // ERROR "go:wasmexport: unsupported parameter type"

//go:wasmexport bad3
func bad3(uint8) {} // ERROR "go:wasmexport: unsupported parameter type"

//go:wasmexport bad4
func bad4(int) {} // ERROR "go:wasmexport: unsupported parameter type"

// Struct and array types are also not allowed.

type S struct { x, y int32 }

type H struct { _ structs.HostLayout; x, y int32 }

type A = structs.HostLayout

type AH struct { _ A; x, y int32 }

//go:wasmexport bad5
func bad5(S) {} // ERROR "go:wasmexport: unsupported parameter type"

//go:wasmexport bad6
func bad6(H) {} // ERROR "go:wasmexport: unsupported parameter type"

//go:wasmexport bad7
func bad7([4]int32) {} // ERROR "go:wasmexport: unsupported parameter type"

// Pointer types are not allowed, with resitrictions on
// the element type.

//go:wasmexport good6
func good6(*int32, *uint8, *bool) {}

//go:wasmexport bad8
func bad8(*S) {} // ERROR "go:wasmexport: unsupported parameter type" // without HostLayout, not allowed

//go:wasmexport bad9
func bad9() *S { return nil } // ERROR "go:wasmexport: unsupported result type"

//go:wasmexport good7
func good7(*H, *AH) {} // pointer to struct with HostLayout is allowed

//go:wasmexport good8
func good8(*struct{}) {} // pointer to empty struct is allowed

//go:wasmexport good9
func good9(*[4]int32, *[2]H) {} // pointer to array is allowed, if the element type is okay

//go:wasmexport toomanyresults
func toomanyresults() (int32, int32) { return 0, 0 } // ERROR "go:wasmexport: too many return values"

//go:wasmexport bad10
func bad10() string { return "" } // ERROR "go:wasmexport: unsupported result type" // string cannot be a result
