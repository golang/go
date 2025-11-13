// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test cases that were invalid because of cycles before the respective language change.
// Some are still invalid, but not because of cycles.

package p

type T1[P T1[P]] struct{}
type T2[P interface {
	T2[int /* ERROR "int does not satisfy interface{T2[int]}" */]
}] struct{}
type T3[P interface {
	m(T3[int /* ERROR "int does not satisfy interface{m(T3[int])}" */])
}] struct{}
type T4[P T5[P /* ERROR "P does not satisfy T4[P]" */]] struct{}
type T5[P T4[P /* ERROR "P does not satisfy T5[P]" */]] struct{}

type T6[P int] struct{ f *T6[P] }
