// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "io"

// test using struct with invalid embedded field
var _ io.Writer = W{} // no error expected here because W has invalid embedded field

type W struct {
	*bufio /* ERROR "undefined: bufio" */ .Writer
}

// test using an invalid type
var _ interface{ m() } = &M{} // no error expected here because M is invalid

type M undefined // ERROR "undefined: undefined"

// test using struct with invalid embedded field and containing a self-reference (cycle)
var _ interface{ m() } = &S{} // no error expected here because S is invalid

type S struct {
	*S
	undefined // ERROR "undefined: undefined"
}

// test using a generic struct with invalid embedded field and containing a self-reference (cycle)
var _ interface{ m() } = &G[int]{} // no error expected here because S is invalid

type G[P any] struct {
	*G[P]
	undefined // ERROR "undefined: undefined"
}
