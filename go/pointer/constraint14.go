// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.5

package pointer

import (
	"golang.org/x/tools/go/types"
)

type constraint interface {
	// For a complex constraint, returns the nodeid of the pointer
	// to which it is attached.   For addr and copy, returns dst.
	ptr() nodeid

	// renumber replaces each nodeid n in the constraint by mapping[n].
	renumber(mapping []nodeid)

	// presolve is a hook for constraint-specific behaviour during
	// pre-solver optimization.  Typical implementations mark as
	// indirect the set of nodes to which the solver will add copy
	// edges or PTS labels.
	presolve(h *hvn)

	// solve is called for complex constraints when the pts for
	// the node to which they are attached has changed.
	solve(a *analysis, delta *nodeset)

	String() string
}

// dst = &src
// pts(dst) ⊇ {src}
// A base constraint used to initialize the solver's pt sets
type addrConstraint struct {
	dst nodeid // (ptr)
	src nodeid
}

func (c *addrConstraint) ptr() nodeid { return c.dst }
func (c *addrConstraint) renumber(mapping []nodeid) {
	c.dst = mapping[c.dst]
	c.src = mapping[c.src]
}

// dst = src
// A simple constraint represented directly as a copyTo graph edge.
type copyConstraint struct {
	dst nodeid // (ptr)
	src nodeid
}

func (c *copyConstraint) ptr() nodeid { return c.dst }
func (c *copyConstraint) renumber(mapping []nodeid) {
	c.dst = mapping[c.dst]
	c.src = mapping[c.src]
}

// dst = src[offset]
// A complex constraint attached to src (the pointer)
type loadConstraint struct {
	offset uint32
	dst    nodeid
	src    nodeid // (ptr)
}

func (c *loadConstraint) ptr() nodeid { return c.src }
func (c *loadConstraint) renumber(mapping []nodeid) {
	c.dst = mapping[c.dst]
	c.src = mapping[c.src]
}

// dst[offset] = src
// A complex constraint attached to dst (the pointer)
type storeConstraint struct {
	offset uint32
	dst    nodeid // (ptr)
	src    nodeid
}

func (c *storeConstraint) ptr() nodeid { return c.dst }
func (c *storeConstraint) renumber(mapping []nodeid) {
	c.dst = mapping[c.dst]
	c.src = mapping[c.src]
}

// dst = &src.f  or  dst = &src[0]
// A complex constraint attached to dst (the pointer)
type offsetAddrConstraint struct {
	offset uint32
	dst    nodeid
	src    nodeid // (ptr)
}

func (c *offsetAddrConstraint) ptr() nodeid { return c.src }
func (c *offsetAddrConstraint) renumber(mapping []nodeid) {
	c.dst = mapping[c.dst]
	c.src = mapping[c.src]
}

// dst = src.(typ)  where typ is an interface
// A complex constraint attached to src (the interface).
// No representation change: pts(dst) and pts(src) contains tagged objects.
type typeFilterConstraint struct {
	typ types.Type // an interface type
	dst nodeid
	src nodeid // (ptr)
}

func (c *typeFilterConstraint) ptr() nodeid { return c.src }
func (c *typeFilterConstraint) renumber(mapping []nodeid) {
	c.dst = mapping[c.dst]
	c.src = mapping[c.src]
}

// dst = src.(typ)  where typ is a concrete type
// A complex constraint attached to src (the interface).
//
// If exact, only tagged objects identical to typ are untagged.
// If !exact, tagged objects assignable to typ are untagged too.
// The latter is needed for various reflect operators, e.g. Send.
//
// This entails a representation change:
// pts(src) contains tagged objects,
// pts(dst) contains their payloads.
type untagConstraint struct {
	typ   types.Type // a concrete type
	dst   nodeid
	src   nodeid // (ptr)
	exact bool
}

func (c *untagConstraint) ptr() nodeid { return c.src }
func (c *untagConstraint) renumber(mapping []nodeid) {
	c.dst = mapping[c.dst]
	c.src = mapping[c.src]
}

// src.method(params...)
// A complex constraint attached to iface.
type invokeConstraint struct {
	method *types.Func // the abstract method
	iface  nodeid      // (ptr) the interface
	params nodeid      // the start of the identity/params/results block
}

func (c *invokeConstraint) ptr() nodeid { return c.iface }
func (c *invokeConstraint) renumber(mapping []nodeid) {
	c.iface = mapping[c.iface]
	c.params = mapping[c.params]
}
