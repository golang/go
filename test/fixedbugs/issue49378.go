// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f(i int) {
	var s1 struct {
		s struct{ s struct{ i int } }
	}
	var s2, s3 struct {
		a struct{ i int }
		b int
	}
	func() {
		i = 1 + 2*i + s3.a.i + func() int {
			s2.a, s2.b = s3.a, s3.b
			return 0
		}() + func(*int) int {
			return s1.s.s.i
		}(new(int))
	}()
}
