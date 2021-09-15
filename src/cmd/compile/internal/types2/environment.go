// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package types2

import (
	"bytes"
	"strings"
	"sync"
)

// An Environment is an opaque type checking environment. It may be used to
// share identical type instances across type-checked packages or calls to
// Instantiate.
//
// It is safe for concurrent use.
type Environment struct {
	mu      sync.Mutex
	typeMap map[string]*Named // type hash -> instance
	nextID  int               // next unique ID
	seen    map[*Named]int    // assigned unique IDs
}

// NewEnvironment creates a new Environment.
func NewEnvironment() *Environment {
	return &Environment{
		typeMap: make(map[string]*Named),
		seen:    make(map[*Named]int),
	}
}

// TypeHash returns a string representation of typ, which can be used as an exact
// type hash: types that are identical produce identical string representations.
// If typ is a *Named type and targs is not empty, typ is printed as if it were
// instantiated with targs. The result is guaranteed to not contain blanks (" ").
func (env *Environment) TypeHash(typ Type, targs []Type) string {
	assert(env != nil)
	assert(typ != nil)
	var buf bytes.Buffer

	h := newTypeHasher(&buf, env)
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

// typeForHash returns the recorded type for the type hash h, if it exists.
// If no type exists for h and n is non-nil, n is recorded for h.
func (env *Environment) typeForHash(h string, n *Named) *Named {
	env.mu.Lock()
	defer env.mu.Unlock()
	if existing := env.typeMap[h]; existing != nil {
		return existing
	}
	if n != nil {
		env.typeMap[h] = n
	}
	return n
}

// idForType returns a unique ID for the pointer n.
func (env *Environment) idForType(n *Named) int {
	env.mu.Lock()
	defer env.mu.Unlock()
	id, ok := env.seen[n]
	if !ok {
		id = env.nextID
		env.seen[n] = id
		env.nextID++
	}
	return id
}
