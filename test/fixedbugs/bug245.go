// $G $D/$F.go || echo BUG: bug245

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T []int
func (t T) m()

func main() {
	_ = T{}
}

// bug245.go:14: fatal error: method mismatch: T for T
