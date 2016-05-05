// cmpout

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that we can defer the predeclared functions print and println.

package main

func main() {
	defer println(42, true, false, true, 1.5, "world", (chan int)(nil), []int(nil), (map[string]int)(nil), (func())(nil), byte(255))
	defer println(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)
	// Disabled so the test doesn't crash but left here for reference.
	// defer panic("dead")
	defer print("printing: ")
}
