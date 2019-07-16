// compile

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7590: gccgo incorrectly traverses nested composite literals.

package p

type S struct {
	F int
}

var M = map[string]S{
	"a": { F: 1 },
}

var P = M["a"]

var F = P.F
