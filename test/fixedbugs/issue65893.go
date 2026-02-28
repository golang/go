// compile

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type (
	s  = struct{ f func(s1) }
	s1 = struct{ i I }
)

type I interface {
	S() *s
}
