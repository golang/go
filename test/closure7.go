// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func g(f func()) {
}

// Must have exportable name
func F() {
	g(func() {
		ch := make(chan int)
		for {
			select {
			case <-ch:
				return
			default:
			}
		}
	})
}

func main() {
	F()
}
