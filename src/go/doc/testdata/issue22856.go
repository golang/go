// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue22856

type T struct{}

func New() T                   { return T{} }
func NewPointer() *T           { return &T{} }
func NewPointerSlice() []*T    { return []*T{&T{}} }
func NewSlice() []T            { return []T{T{}} }
func NewPointerOfPointer() **T { x := &T{}; return &x }
func NewArray() [1]T           { return [1]T{T{}} }
func NewPointerArray() [1]*T   { return [1]*T{&T{}} }

// NewSliceOfSlice is not a factory function because slices of a slice of
// type *T are not factory functions of type T.
func NewSliceOfSlice() [][]T { return []T{[]T{}} }

// NewPointerSliceOfSlice is not a factory function because slices of a
// slice of type *T are not factory functions of type T.
func NewPointerSliceOfSlice() [][]*T { return []*T{[]*T{}} }

// NewSlice3 is not a factory function because 3 nested slices of type T
// are not factory functions of type T.
func NewSlice3() [][][]T { return []T{[]T{[]T{}}} }
