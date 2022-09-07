// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

// The actual example from the issue.
type List[P any] struct{}

func (_ List[P]) m() (_ List[List[P]]) { return }

// Other types of recursion through methods.
type R[P any] int

func (*R[R /* ERROR must be an identifier */ [int]]) m0() {}
func (R[P]) m1(R[R[P]])                                   {}
func (R[P]) m2(R[*P])                                     {}
func (R[P]) m3([unsafe.Sizeof(new(R[P]))]int)             {}
func (R[P]) m4([unsafe.Sizeof(new(R[R[P]]))]int)          {}

// Mutual recursion
type M[P any] int

func (R[P]) m5(M[M[P]]) {}
func (M[P]) m(R[R[P]])  {}
