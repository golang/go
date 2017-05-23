// compile

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Crashed gccgo.

package p

type S struct {
	f interface{}
}

func F(p *S) bool {
	v := p.f
	switch a := v.(type) {
	case nil:
		_ = a
		return true
	}
	return true
}
