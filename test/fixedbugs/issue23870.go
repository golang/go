// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Crashed gccgo.

package p

var F func() [0]struct{
	A int
}

var i int
var V = (F()[i]).A
