// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f[T int](t T) {
	for true {
		func() {
			t = func() T { return t }()
		}()
	}
}

func main() {
	f(0)
}
