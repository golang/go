// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "sync"

// methodList holds a list of methods that may be lazily resolved by a provided
// resolution method.
type methodList struct {
	methods []*Func

	// guards synchronizes the instantiation of lazy methods. For lazy method
	// lists, guards is non-nil and of the length passed to newLazyMethodList.
	// For non-lazy method lists, guards is nil.
	guards *[]sync.Once
}

// newMethodList creates a non-lazy method list holding the given methods.
func newMethodList(methods []*Func) *methodList {
	return &methodList{methods: methods}
}

// newLazyMethodList creates a lazy method list of the given length. Methods
// may be resolved lazily for a given index by providing a resolver function.
func newLazyMethodList(length int) *methodList {
	guards := make([]sync.Once, length)
	return &methodList{
		methods: make([]*Func, length),
		guards:  &guards,
	}
}

// isLazy reports whether the receiver is a lazy method list.
func (l *methodList) isLazy() bool {
	return l != nil && l.guards != nil
}

// Add appends a method to the method list if not not already present. Add
// panics if the receiver is lazy.
func (l *methodList) Add(m *Func) {
	assert(!l.isLazy())
	if i, _ := lookupMethod(l.methods, m.pkg, m.name, false); i < 0 {
		l.methods = append(l.methods, m)
	}
}

// Lookup looks up the method identified by pkg and name in the receiver.
// Lookup panics if the receiver is lazy. If foldCase is true, method names
// are considered equal if they are equal with case folding.
func (l *methodList) Lookup(pkg *Package, name string, foldCase bool) (int, *Func) {
	assert(!l.isLazy())
	if l == nil {
		return -1, nil
	}
	return lookupMethod(l.methods, pkg, name, foldCase)
}

// Len returns the length of the method list.
func (l *methodList) Len() int {
	if l == nil {
		return 0
	}
	return len(l.methods)
}

// At returns the i'th method of the method list. At panics if i is out of
// bounds, or if the receiver is lazy and resolve is nil.
func (l *methodList) At(i int, resolve func() *Func) *Func {
	if !l.isLazy() {
		return l.methods[i]
	}
	assert(resolve != nil)
	(*l.guards)[i].Do(func() {
		l.methods[i] = resolve()
	})
	return l.methods[i]
}
