// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !compiler_bootstrap

package test

// The racecompile builder only builds packages, but does not build
// or run tests. This is a non-test file to hold cases that (used
// to) trigger compiler data races, so they will be exercised on
// the racecompile builder.
//
// This package is not imported so functions here are not included
// in the actual compiler.

// Issue 55357: data race when building multiple instantiations of
// generic closures with _ parameters.
func Issue55357() {
	type U struct {
		A int
		B string
		C string
	}
	var q T55357[U]
	q.Count()
	q.List()

	type M struct {
		A int64
		B uint32
		C uint32
	}
	var q2 T55357[M]
	q2.Count()
	q2.List()
}

type T55357[T any] struct{}

//go:noinline
func (q *T55357[T]) do(w, v bool, fn func(bk []byte, v T) error) error {
	return nil
}

func (q *T55357[T]) Count() (n int, rerr error) {
	err := q.do(false, false, func(kb []byte, _ T) error {
		n++
		return nil
	})
	return n, err
}

func (q *T55357[T]) List() (list []T, rerr error) {
	var l []T
	err := q.do(false, true, func(_ []byte, v T) error {
		l = append(l, v)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return l, nil
}
