// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure SSA->assembly pass can handle SP as an index register.

package p

type T struct {
	a,b,c,d float32
}

func f(a *[8]T, i,j,k int) float32 {
	b := *a
	return b[i].a + b[j].b + b[k].c
}
