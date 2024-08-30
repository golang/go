// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	defer func() {
		println(recover().(int))
	}()
	func() {
		func() (_ [2]int) { type _ int; return }()
		func() {
			defer func() {
				defer func() {
					recover()
				}()
				defer panic(3)
				panic(2)
			}()
			defer func() {
				recover()
			}()
			panic(1)
		}()
		defer func() {}()
	}()

	var x = 123
	func() {
		// in the original issue, this defer was not executed (which is incorrect)
		defer print(x)
		func() {
			defer func() {}()
			panic(4)
		}()
	}()
}
