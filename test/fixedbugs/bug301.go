// $G $D/$F.go || echo BUG: bug301.go

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// http://code.google.com/p/go/issues/detail?id=990

package main

func main() {
	defer func() {
		if recover() != nil {
			panic("non-nil recover")
		}
	}()
	panic(nil)
}
