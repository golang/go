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

// Index provides lookup of concrete spec functions by (receiver, name), and of
// API types by name.
type Index struct {
	m     map[indexKey]*Func
	funcs []*Func

	typeMap map[string]*TypeDecl
	types   []*TypeDecl
}

// NewIndex builds an Index over the given spec functions and API types.
func NewIndex(funcs []*Func, types []*TypeDecl) *Index {
	m := make(map[indexKey]*Func, len(funcs))
	for _, fn := range funcs {
		var recv string
		if fn.Recv.Type != nil {
			recv = fn.Recv.Type.String()
		}
		m[indexKey{recv: recv, name: fn.Name}] = fn
	}
	typeMap := make(map[string]*TypeDecl, len(types))
	for _, t := range types {
		typeMap[t.Name] = t
	}
	return &Index{
		m:       m,
		funcs:   funcs,
		typeMap: typeMap,
		types:   types,
	}
}

// LoadIndex loads a Go SIMD spec from the package in directory dir, like
// [Load], and returns an [Index] over it.
func LoadIndex(dir string, opts *LoadOptions) (*Index, error) {
	funcs, types, err := Load(dir, opts)
	if err != nil {
		return nil, err
	}
	return NewIndex(funcs, types), nil
}

// Lookup finds the spec function for the given receiver and function/method name.
// For package-level functions, recv should be "".
func (idx *Index) Lookup(recv, name string) *Func {
	if idx == nil {
		return nil
	}
	return idx.m[indexKey{recv: recv, name: name}]
}

// LookupType finds the API type with the given name, or returns nil.
func (idx *Index) LookupType(name string) *TypeDecl {
	if idx == nil {
		return nil
	}
	return idx.typeMap[name]
}

// Funcs returns all functions in the index.
func (idx *Index) Funcs() []*Func {
	if idx == nil {
		return nil
	}
	return idx.funcs
}

// Types returns all API types in the index.
func (idx *Index) Types() []*TypeDecl {
	if idx == nil {
		return nil
	}
	return idx.types
}

// Len returns the number of functions in the index.
func (idx *Index) Len() int {
	if idx == nil {
		return 0
	}
	return len(idx.m)
}
