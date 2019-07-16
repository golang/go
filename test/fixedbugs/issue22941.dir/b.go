// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import q "./a"

type T struct {
	X *q.P
}

func F(in, out *T) {
	*out = *in
	if in.X != nil {
		in, out := &in.X, &out.X
		if *in == nil {
			*out = nil
		} else {
			*out = new(q.P)
			**out = **in
		}
	}
	return
}

//go:noinline
func G(x, y *T) {
	F(x, y)
}
