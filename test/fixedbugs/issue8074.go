// compile

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 8074.
// was "cannot take the address of 1"

package main

func main() {
	a := make([]byte, 10)
	m := make(map[float64][]byte)
	go copy(a, m[1.0])
}
