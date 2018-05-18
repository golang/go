// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytealg

// Note: there's no equal_generic.go because every platform must implement at least memequal_varlen in assembly.

//go:noescape
func Equal(a, b []byte) bool

// The compiler generates calls to runtime.memequal and runtime.memequal_varlen.
// In addition, the runtime calls runtime.memequal explicitly.
// Those functions are implemented in this package.
