// compile

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Gccgo used to crash compiling this.

package p

type E int

func (e E) P() *E { return &e }

const (
	C1 E = 0
	C2 = C1
)

func F() *E {
	return C2.P()
}
