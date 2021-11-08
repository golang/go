// compile

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var n, a, b int64
	for i := int64(2); i < 10; i++ {
		for j := i; j < 10; j++ {
			if ((n % (i * j)) == 0) && (j > 1 && (n/(i*j)) == 1) {
				a, b = i, 0
				a = n / (i * j)
			}
		}
	}

	if a != b && a != n {
		println("yes")
	}
}
