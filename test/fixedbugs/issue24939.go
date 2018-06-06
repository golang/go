// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T interface {
	M(P)
}

type M interface {
	F() P
}

type P = interface {
	I() M
}

func main() {}
