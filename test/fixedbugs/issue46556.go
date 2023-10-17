// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type A = interface{}
type B interface{}

// Test that embedding both anonymous and defined types is supported.
type C interface {
	A
	B
}
