// $G $D/$F.go || echo BUG: bug220

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	m := make(map[int]map[uint]float64)

	m[0] = make(map[uint]float64), false // 6g used to reject this
	m[1] = nil
}
