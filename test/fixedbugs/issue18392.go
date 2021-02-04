// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type A interface {
	// TODO(mdempsky): This should be an error, but this error is
	// nonsense. The error should actually mention that there's a
	// type loop.
	Fn(A.Fn) // ERROR "type A has no method Fn|A.Fn undefined"
}
