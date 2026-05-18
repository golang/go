// run -goexperiment genericmethods

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that generic methods are assembled correctly.

package main

import "fmt"

type List[E any] []E

func (l List[E]) apply[P any](f func(E) P) List[P] {
	r := make(List[P], len(l))
	for i, x := range l {
		r[i] = f(x)
	}
	return r
}

func (l List[E]) print() string {
	s := ""
	for i, x := range l {
		if i > 0 {
			s += "->"
		}
		s += fmt.Sprint(x)
	}
	return s
}

func main() {
	l := make(List[int], 3)
	for i := range l {
		l[i] = i
	}

	f := func(i int) string {
		return string("abc"[i])
	}

	if r := l.apply(f).print(); r != "a->b->c" {
		panic(fmt.Sprintf("got %s, want a->b->c", r))
	}
}
