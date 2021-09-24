// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
//go:build go1.18
// +build go1.18

package testdata

type List[E any] []E

// TODO(suzmue): add a test for generic slice expressions when https://github.com/golang/go/issues/48618 is closed.
// type S interface{ ~[]int }

var (
	a [10]byte
	b [20]float32
	p List[int]

	_ = p[0:]
	_ = p[1:10]
	_ = p[2:] // want "unneeded: len\\(p\\)"
	_ = p[3:(len(p))]
	_ = p[len(a) : len(p)-1]
	_ = p[0:len(b)]
	_ = p[2:len(p):len(p)]

	_ = p[:]
	_ = p[:10]
	_ = p[:] // want "unneeded: len\\(p\\)"
	_ = p[:(len(p))]
	_ = p[:len(p)-1]
	_ = p[:len(b)]
	_ = p[:len(p):len(p)]
)

func foo[E any](a List[E]) {
	_ = a[0:] // want "unneeded: len\\(a\\)"
}
