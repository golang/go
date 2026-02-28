// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Composite literals with parameterized types

package comp_literals

type myStruct struct {
	f int
}

type slice[E any] []E

func struct_literals[S struct{f int}|myStruct]() {
	_ = S{}
	_ = S{0}
	_ = S{f: 0}

        _ = slice[int]{1, 2, 3}
        _ = slice[S]{{}, {0}, {f:0}}
}
