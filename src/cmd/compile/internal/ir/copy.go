// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/internal/src"
)

// Copy returns a shallow copy of n.
func Copy(n Node) Node {
	return n.copy()
}

// DeepCopy returns a “deep” copy of n, with its entire structure copied
// (except for shared nodes like ONAME, ONONAME, OLITERAL, and OTYPE).
// If pos.IsKnown(), it sets the source position of newly allocated Nodes to pos.
func DeepCopy(pos src.XPos, n Node) Node {
	var edit func(Node) Node
	edit = func(x Node) Node {
		switch x.Op() {
		case ONAME, ONONAME, OLITERAL, ONIL, OTYPE:
			return x
		}
		x = Copy(x)
		if pos.IsKnown() {
			x.SetPos(pos)
		}
		EditChildren(x, edit)
		return x
	}
	return edit(n)
}
