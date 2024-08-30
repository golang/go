// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// gofrontend miscompiled some cases of append(s, make(typ, ln)...).

package main

var g int

func main() {
	a := []*int{&g, &g, &g, &g}
	a = append(a[:0], make([]*int, len(a) - 1)...)
	if len(a) != 3 || a[0] != nil || a[1] != nil || a[2] != nil {
		panic(a)
	}
}
