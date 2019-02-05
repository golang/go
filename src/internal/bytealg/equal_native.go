// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytealg

import "unsafe"

// Note: there's no equal_generic.go because every platform must implement at least memequal_varlen in assembly.

//go:noescape
func Equal(a, b []byte) bool

// The declarations below generate ABI wrappers for functions
// implemented in assembly in this package but declared in another
// package.

// The compiler generates calls to runtime.memequal and runtime.memequal_varlen.
// In addition, the runtime calls runtime.memequal explicitly.
// Those functions are implemented in this package.

//go:linkname abigen_runtime_memequal runtime.memequal
func abigen_runtime_memequal(a, b unsafe.Pointer, size uintptr) bool

//go:linkname abigen_runtime_memequal_varlen runtime.memequal_varlen
func abigen_runtime_memequal_varlen(a, b unsafe.Pointer) bool
