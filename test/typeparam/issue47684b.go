// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f[G any]() interface{} {
	return func() interface{} {
		return func() interface{} {
			var x G
			return x
		}()
	}()
}

func main() {
	x := f[int]()
	if v, ok := x.(int); !ok || v != 0 {
		panic("bad")
	}
}
