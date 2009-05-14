// $G $D/$F.go || echo BUG: bug151

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S string

type Empty interface {}

func (v S) Less(e Empty) bool {
	return v < e.(S);
}

/*
bugs/bug151.go:10: illegal types for operand: CALL
	string
	S
*/
