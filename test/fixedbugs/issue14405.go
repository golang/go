// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Mention of field with large offset in struct literal causes crash
package p

type T struct {
	Slice [1 << 20][]int
	Ptr   *int
}

func New(p *int) *T {
	return &T{Ptr: p}
}
