// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f1[P interface{ *P }]() {}
func f2[P interface{ func(P) }]() {}
func f3[P, Q interface{ func(Q) P }]() {}
func f4[P interface{ *Q }, Q interface{ func(P) }]() {}
func f5[P interface{ func(P) }]() {}
func f6[P interface { *Tree[P] }, Q any ]() {}

func _() {
        f1( /* ERROR cannot infer P */ )
        f2( /* ERROR cannot infer P */ )
        f3( /* ERROR cannot infer P */ )
        f4( /* ERROR cannot infer P */ )
        f5( /* ERROR cannot infer P */ )
        f6( /* ERROR cannot infer P */ )
}

type Tree[P any] struct {
        left, right *Tree[P]
        data P
}

// test case from issue

func foo[Src interface { func() Src }]() Src {
        return foo[Src]
}

func _() {
        foo( /* ERROR cannot infer Src */ )
}
