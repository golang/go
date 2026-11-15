// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type Node struct {
	Next *Node
}

//go:noinline
func Link(o, p *Node) { // ERROR "o does not escape" "p does not escape"
	o.Next = p
}
