// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test simple switch.

package main

func main() {
	r := ""
	a := 3
	for i := 0; i < 10; i = i + 1 {
		switch i {
		case 5:
			r += "five"
		case a, 7:
			r += "a"
		default:
			r += string(i + '0')
		}
		r += "out" + string(i+'0')
	}
	if r != "0out01out12out2aout34out4fiveout56out6aout78out89out9" {
		panic(r)
	}
}
