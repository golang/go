// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $D/$F.go || (echo BUG: should compile cleanly; exit 1)
package main

func f() int {
	i := 0
	for {
		if i >= sys.argc() {
			return i
		}
		i++
	}
}

func g() int {
	for {
	}
}

func h() int {
	for {
		return 1
	}
}
