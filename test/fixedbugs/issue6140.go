// compile

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 6140: compiler incorrectly rejects method values
// whose receiver has an unnamed interface type.

package p

type T *interface {
	m() int
}

var x T

var _ = (*x).m

var y interface {
	m() int
}

var _ = y.m

type I interface {
	String() string
}

var z *struct{ I }
var _ = z.String
