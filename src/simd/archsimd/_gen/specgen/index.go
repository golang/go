// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

// indexKey identifies an API function or method by receiver type name and
// method name. For package-level functions, recv is empty.
type indexKey struct {
	recv string
	name string
}

// Index provides lookup of concrete spec functions by (receiver, name).
type Index struct {
	m     map[indexKey]*Func
	funcs []*Func
}

// NewIndex builds an Index over the given slice of spec functions.
func NewIndex(funcs []*Func) *Index {
	m := make(map[indexKey]*Func, len(funcs))
	for _, fn := range funcs {
		var recv string
		if fn.Recv.Type != nil {
			recv = fn.Recv.Type.String()
		}
		m[indexKey{recv: recv, name: fn.Name}] = fn
	}
	return &Index{
		m:     m,
		funcs: funcs,
	}
}

// Lookup finds the spec function for the given receiver and function/method name.
// For package-level functions, recv should be "".
func (idx *Index) Lookup(recv, name string) *Func {
	if idx == nil {
		return nil
	}
	return idx.m[indexKey{recv: recv, name: name}]
}

// Funcs returns all functions in the index.
func (idx *Index) Funcs() []*Func {
	if idx == nil {
		return nil
	}
	return idx.funcs
}

// Len returns the number of functions in the index.
func (idx *Index) Len() int {
	if idx == nil {
		return 0
	}
	return len(idx.m)
}
