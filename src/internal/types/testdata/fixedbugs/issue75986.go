// -lang=go1.25

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "strings"

type T int
type G[P any] struct{}

var x T

// Verify that we don't get a version error when there's another error present in new(expr).

func f() {
	_ = new(U /* ERROR "undefined: U" */)
	_ = new(strings.BUILDER /* ERROR "undefined: strings.BUILDER (but have Builder)" */)
	_ = new(T)      // ok
	_ = new(G[int]) // ok
	_ = new(G /* ERROR "cannot use generic type G without instantiation" */)
	_ = new(nil /* ERROR "use of untyped nil in argument to new" */)
	_ = new(comparable /* ERROR "cannot use type comparable outside a type constraint" */)
	_ = new(new /* ERROR "new (built-in) must be called" */)
	_ = new(panic /* ERROR "panic(0) (no value) used as value or type" */ (0))
}
