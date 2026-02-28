// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	env := func() func(*bool) func() int {
		return func() func(*bool) func() int {
			return func(ptr *bool) func() int {
				return func() int {
					*ptr = true
					return 0
				}
			}
		}()
	}()

	var ok bool
	func(int) {}(env(&ok)())
	if !ok {
		panic("FAIL")
	}
}
