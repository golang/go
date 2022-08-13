// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//gofmt

package typeparams

type T[  P any] struct{}
type T[P1, P2, P3 any] struct{}

type T[P C] struct{}
type T[P1,P2, P3 C] struct{}

type T[P C[P]] struct{}
type T[P1, P2, P3 C[P1,P2,P3]] struct{}

func f[P any](x P)
func f[P1, P2, P3 any](x1 P1, x2 P2, x3 P3) struct{}

func f[P interface{}](x P)
func f[P1, P2, P3 interface{ m1(P1); ~P2|~P3 }](x1 P1, x2 P2, x3 P3) struct{}
func f[P any](T1[P], T2[P]) T3[P]

func (x T[P]) m()
func ((T[P])) m(x T[P]) P

func _() {
	type _ []T[P]
	var _ []T[P]
	_ = []T[P]{}
}
