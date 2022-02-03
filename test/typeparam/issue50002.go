// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test for cases where certain instantiations of a generic function (F in this
// example) will always fail on a type assertion or mismatch on a type case.

package main

import "fmt"

type S struct{}

func (S) M() byte {
	return 0
}

type I[T any] interface {
	M() T
}

func F[T, A any](x I[T], shouldMatch bool) {
	switch x.(type) {
	case A:
		if !shouldMatch {
			fmt.Printf("wanted mis-match, got match")
		}
	default:
		if shouldMatch {
			fmt.Printf("wanted match, got mismatch")
		}
	}

	_, ok := x.(A)
	if ok != shouldMatch {
		fmt.Printf("ok: got %v, wanted %v", ok, shouldMatch)
	}

	if !shouldMatch {
		defer func() {
			if shouldMatch {
				fmt.Printf("Shouldn't have panicked")
			}
			recover()
		}()
	}
	_ = x.(A)
	if !shouldMatch {
		fmt.Printf("Should have panicked")
	}
}

func main() {
	// Test instantiation where the type switch/type asserts can't possibly succeed
	// (since string does not implement I[byte]).
	F[byte, string](S{}, false)

	// Test instantiation where the type switch/type asserts should succeed
	// (since S does implement I[byte])
	F[byte, S](S{}, true)
	F[byte, S](I[byte](S{}), true)
}
