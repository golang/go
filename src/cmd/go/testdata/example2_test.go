// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure that go test runs Example_Y before Example_B, preserving source order.

package p

import "fmt"

func Example_Y() {
	n++
	fmt.Println(n)
	// Output: 3
}

func Example_B() {
	n++
	fmt.Println(n)
	// Output: 4
}
