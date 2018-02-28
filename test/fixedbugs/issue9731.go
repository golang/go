// compile

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f(x interface{}) {
	switch x := x.(type) {
	case int:
		func() {
			_ = x
		}()
	case map[int]int:
		func() {
			for range x {
			}
		}()
	}
}
