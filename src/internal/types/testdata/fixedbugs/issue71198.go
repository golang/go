// -gotypesalias=1

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type A[_ any] = any

// This must not panic; also the error message must match the style for non-alias types, below.
func _[_ A /* ERROR "too many type arguments for type A: have 2, want 1" */ [int, string]]() {}

type T[_ any] any

func _[_ T /* ERROR "too many type arguments for type T: have 2, want 1" */ [int, string]]() {}
