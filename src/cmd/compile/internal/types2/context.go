// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package types2

import (
	"bytes"
	"strings"
	"sync"
)

// An Context is an opaque type checking context. It may be used to share
// identical type instances across type-checked packages or calls to
// Instantiate.
//
// It is safe for concurrent use.
type Context struct {
	mu      sync.Mutex
	typeMap map[string]*Named // type hash -> instance
	nextID  int               // next unique ID
	seen    map[*Named]int    // assigned unique IDs
}

// NewContext creates a new Context.
func NewContext() *Context {
	return &Context{
		typeMap: make(map[string]*Named),
		seen:    make(map[*Named]int),
	}
}

// TypeHash returns a string representation of typ, which can be used as an exact
// type hash: types that are identical produce identical string representations.
// If typ is a *Named type and targs is not empty, typ is printed as if it were
// instantiated with targs. The result is guaranteed to not contain blanks (" ").
func (ctxt *Context) TypeHash(typ Type, targs []Type) string {
	assert(ctxt != nil)
	assert(typ != nil)
	var buf bytes.Buffer

	h := newTypeHasher(&buf, ctxt)
	if named := asNamed(typ); named != nil && len(targs) > 0 {
		// Don't use WriteType because we need to use the provided targs
		// and not any targs that might already be with the *Named type.
		h.typePrefix(named)
		h.typeName(named.obj)
		h.typeList(targs)
	} else {
		assert(targs == nil)
		h.typ(typ)
	}

	return strings.Replace(buf.String(), " ", "#", -1) // ReplaceAll is not available in Go1.4
}

// typeForHash returns the recorded type for the type hash h, if it exists.
// If no type exists for h and n is non-nil, n is recorded for h.
func (ctxt *Context) typeForHash(h string, n *Named) *Named {
	ctxt.mu.Lock()
	defer ctxt.mu.Unlock()
	if existing := ctxt.typeMap[h]; existing != nil {
		return existing
	}
	if n != nil {
		ctxt.typeMap[h] = n
	}
	return n
}

// idForType returns a unique ID for the pointer n.
func (ctxt *Context) idForType(n *Named) int {
	ctxt.mu.Lock()
	defer ctxt.mu.Unlock()
	id, ok := ctxt.seen[n]
	if !ok {
		id = ctxt.nextID
		ctxt.seen[n] = id
		ctxt.nextID++
	}
	return id
}
