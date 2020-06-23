// errorcheck -0 -m -l

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for self assignments.

package escape

type S struct {
	i  int
	pi *int
}

var sink S

func f(p *S) { // ERROR "leaking param: p"
	p.pi = &p.i
	sink = *p
}

// BAD: "leaking param: p" is too conservative
func g(p *S) { // ERROR "leaking param: p"
	p.pi = &p.i
}

func h() {
	var s S // ERROR "moved to heap: s"
	g(&s)
	sink = s
}
