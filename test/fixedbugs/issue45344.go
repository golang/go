// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 45344: expand_calls does not handle direct interface
// typed argument well.

package p

type T struct {
	a map[int]int
}

func F(t T) {
	G(t)
}

func G(...interface{})
