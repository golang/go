// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"bytes"
	"fmt"
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
	typeMap map[string][]ctxtEntry // type hash -> instances entries
	nextID  int                    // next unique ID
	seen    map[*Named]int         // assigned unique IDs
}

type ctxtEntry struct {
	orig     Type
	targs    []Type
	instance Type // = orig[targs]
}

// NewContext creates a new Context.
func NewContext() *Context {
	return &Context{
		typeMap: make(map[string][]ctxtEntry),
		seen:    make(map[*Named]int),
	}
}

// typeHash returns a string representation of typ, which can be used as an exact
// type hash: types that are identical produce identical string representations.
// If typ is a *Named type and targs is not empty, typ is printed as if it were
// instantiated with targs. The result is guaranteed to not contain blanks (" ").
func (ctxt *Context) typeHash(typ Type, targs []Type) string {
	assert(ctxt != nil)
	assert(typ != nil)
	var buf bytes.Buffer

	h := newTypeHasher(&buf, ctxt)
	// Caution: don't use asNamed here. TypeHash may be called for unexpanded
	// types. We don't need anything other than name and type arguments below,
	// which do not require expansion.
	if named, _ := typ.(*Named); named != nil && len(targs) > 0 {
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

// lookup returns an existing instantiation of orig with targs, if it exists.
// Otherwise, it returns nil.
func (ctxt *Context) lookup(h string, orig *Named, targs []Type) Type {
	ctxt.mu.Lock()
	defer ctxt.mu.Unlock()

	for _, e := range ctxt.typeMap[h] {
		if identicalInstance(orig, targs, e.orig, e.targs) {
			return e.instance
		}
		if debug {
			// Panic during development to surface any imperfections in our hash.
			panic(fmt.Sprintf("non-identical instances: (orig: %s, targs: %v) and %s", orig, targs, e.instance))
		}
	}

	return nil
}

// update de-duplicates n against previously seen types with the hash h.  If an
// identical type is found with the type hash h, the previously seen type is
// returned. Otherwise, n is returned, and recorded in the Context for the hash
// h.
func (ctxt *Context) update(h string, orig Type, targs []Type, inst Type) Type {
	assert(inst != nil)

	ctxt.mu.Lock()
	defer ctxt.mu.Unlock()

	for _, e := range ctxt.typeMap[h] {
		if inst == nil || Identical(inst, e.instance) {
			return e.instance
		}
		if debug {
			// Panic during development to surface any imperfections in our hash.
			panic(fmt.Sprintf("%s and %s are not identical", inst, e.instance))
		}
	}

	ctxt.typeMap[h] = append(ctxt.typeMap[h], ctxtEntry{
		orig:     orig,
		targs:    targs,
		instance: inst,
	})

	return inst
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
