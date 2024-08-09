// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// MethodsByName sorts methods by name.
type MethodsByName []*Field

func (x MethodsByName) Len() int           { return len(x) }
func (x MethodsByName) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x MethodsByName) Less(i, j int) bool { return x[i].Sym.Less(x[j].Sym) }

// MethodsByNameCmp sorts methods by name.
func MethodsByNameCmp(x, y *Field) int {
	if x.Sym.Less(y.Sym) {
		return -1
	}
	return +1
}

// EmbeddedsByName sorts embedded types by name.
type EmbeddedsByName []*Field

func (x EmbeddedsByName) Len() int           { return len(x) }
func (x EmbeddedsByName) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x EmbeddedsByName) Less(i, j int) bool { return x[i].Type.Sym().Less(x[j].Type.Sym()) }
