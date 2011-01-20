// $G $D/$F.go || echo BUG: bug315

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 1368.

package main

func main() {
	a := complex(2, 2)
	a /= 2
}

/*
bug315.go:13: internal compiler error: optoas: no entry DIV-complex
*/
