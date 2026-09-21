// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type B *struct{ A }
type A interface{ m(B) }

type s struct{}

func (s) m(b B) {}

func main() {
	var b B = new(struct{ A })
	b.A = s{}
	(*b).m(b)
}
