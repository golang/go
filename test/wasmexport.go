// errorcheck

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that misplaced directives are diagnosed.

//go:build wasm

package p

//go:wasmexport F
func F() {} // OK

//go:wasmexport
func G() {} // OK

type S int32

//go:wasmexport M
func (S) M() {} // ERROR "cannot use //go:wasmexport on a method"
