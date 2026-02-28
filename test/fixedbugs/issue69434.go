// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"iter"
)

func All() iter.Seq[int] {
	return func(yield func(int) bool) {
		for i := 0; i < 10; i++ {
			growStack(512)
			if !yield(i) {
				return
			}
		}
	}
}

type S struct {
	round int
}

func NewS(round int) *S {
	s := &S{round: round}
	return s
}

func (s *S) check(round int) {
	if s.round != round {
		panic("bad round")
	}
}

func f() {
	rounds := 0
	s := NewS(rounds)
	s.check(rounds)

	for range All() {
		s.check(rounds)
		rounds++
		s = NewS(rounds)
		s.check(rounds)
	}
}

func growStack(i int) {
	if i == 0 {
		return
	}
	growStack(i - 1)
}

func main() {
	f()
}
