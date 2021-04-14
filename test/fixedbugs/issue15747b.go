// compile

// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 15747: If an ODCL is dropped, for example when inlining,
// then it's easy to end up not initializing the '&x' pseudo-variable
// to point to an actual allocation. The liveness analysis will detect
// this and abort the computation, so this test just checks that the
// compilation succeeds.

package p

type R [100]byte

func (x R) New() *R {
	return &x
}
