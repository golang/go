// errorcheck -0 -l -m

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 21709: range expression overly escapes.

package p

type S struct{}

func (s *S) Inc() {} // ERROR "s does not escape"
var N int

func F1() {
	var s S
	for i := 0; i < N; i++ {
		fs := []func(){ // ERROR "\[\]func\(\){...} does not escape"
			s.Inc, // ERROR "s.Inc does not escape"
		}
		for _, f := range fs {
			f()
		}
	}
}

func F2() {
	var s S
	for i := 0; i < N; i++ {
		for _, f := range []func(){ // ERROR "\[\]func\(\){...} does not escape"
			s.Inc, // ERROR "s.Inc does not escape"
		} {
			f()
		}
	}
}
