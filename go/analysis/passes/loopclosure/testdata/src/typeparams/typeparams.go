// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the loopclosure checker.

//go:build go1.18

package typeparams

import "golang.org/x/sync/errgroup"

func f[T any](data T) {
	print(data)
}

func _[T any]() {
	var s []T
	for i, v := range s {
		go func() {
			f(i) // want "loop variable i captured by func literal"
			f(v) // want "loop variable v captured by func literal"
		}()
	}
}

func loop[P interface{ Go(func() error) }](grp P) {
	var s []int
	for i, v := range s {
		// The checker only matches on methods "(*...errgroup.Group).Go".
		grp.Go(func() error {
			print(i)
			print(v)
			return nil
		})
	}
}

func _() {
	g := new(errgroup.Group)
	loop(g) // the analyzer is not "type inter-procedural" so no findings are reported
}

type T[P any] struct {
	a P
}

func (t T[P]) Go(func() error) { }

func _(g T[errgroup.Group]) {
	var s []int
	for i, v := range s {
		// "T.a" is method "(*...errgroup.Group).Go".
		g.a.Go(func() error {
			print(i)  // want "loop variable i captured by func literal"
			print(v)  // want "loop variable v captured by func literal"
			return nil
		})
	}
}