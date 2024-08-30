// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var V []int

func f(i int, c chan int) int {
	arr := []int{0, 1}
	for range c {
		for a2 := range arr {
			var a []int
			V = V[:1/a2]
			a[i] = 0
		}
		return func() int {
			arr = []int{}
			return func() int {
				return func() int {
					return func() int { return 4 }()
				}()
			}()
		}()
	}

	return 0
}
