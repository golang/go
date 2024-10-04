// errorcheck

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that wasmexport supports allowed types and rejects
// unallowed types.

//go:build wasm

package p

import "unsafe"

//go:wasmexport good1
func good1(int32, uint32, int64, uint64, float32, float64, unsafe.Pointer) {} // allowed types

type MyInt32 int32

//go:wasmexport good2
func good2(MyInt32) {} // named type is ok

//go:wasmexport good3
func good3() int32 { return 0 } // one result is ok

//go:wasmexport good4
func good4() unsafe.Pointer { return nil } // one result is ok

//go:wasmexport bad1
func bad1(string) {} // ERROR "go:wasmexport: unsupported parameter type"

//go:wasmexport bad2
func bad2(any) {} // ERROR "go:wasmexport: unsupported parameter type"

//go:wasmexport bad3
func bad3(func()) {} // ERROR "go:wasmexport: unsupported parameter type"

//go:wasmexport bad4
func bad4(uint8) {} // ERROR "go:wasmexport: unsupported parameter type"

// Pointer types are not allowed, except unsafe.Pointer.
// Struct and array types are also not allowed.
// If proposal 66984 is accepted and implemented, we may allow them.

//go:wasmexport bad5
func bad5(*int32) {} // ERROR "go:wasmexport: unsupported parameter type"

type S struct { x, y int32 }

//go:wasmexport bad6
func bad6(S) {} // ERROR "go:wasmexport: unsupported parameter type"

//go:wasmexport bad7
func bad7(*S) {} // ERROR "go:wasmexport: unsupported parameter type"

//go:wasmexport bad8
func bad8([4]int32) {} // ERROR "go:wasmexport: unsupported parameter type"

//go:wasmexport bad9
func bad9() bool { return false } // ERROR "go:wasmexport: unsupported result type"

//go:wasmexport bad10
func bad10() *byte { return nil } // ERROR "go:wasmexport: unsupported result type"

//go:wasmexport toomanyresults
func toomanyresults() (int32, int32) { return 0, 0 } // ERROR "go:wasmexport: too many return values"
