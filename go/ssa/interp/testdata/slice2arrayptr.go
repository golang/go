// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Test for slice to array pointer conversion introduced in go1.17

import "fmt"

var s = []byte{1, 2, 3, 4}
var a = (*[4]byte)(s)

func main() {
	for i := range s {
		if a[i] != s[i] {
			panic(fmt.Sprintf("value mismatched: %v - %v\n", a[i], s[i]))
		}
		if (*a)[i] != s[i] {
			panic(fmt.Sprintf("value mismatched: %v - %v\n", (*a)[i], s[i]))
		}
	}
}
