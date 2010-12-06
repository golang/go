// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"go/token"
	"log"
)

/*
 * Blocks and scopes
 */

// A definition can be a *Variable, *Constant, or Type.
type Def interface {
	Pos() token.Pos
}

type Variable struct {
	VarPos token.Pos
	// Index of this variable in the Frame structure
	Index int
	// Static type of this variable
	Type Type
	// Value of this variable.  This is only used by Scope.NewFrame;
	// therefore, it is useful for global scopes but cannot be used
	// in function scopes.
	Init Value
}

func (v *Variable) Pos() token.Pos {
	return v.VarPos
}

type Constant struct {
	ConstPos token.Pos
	Type     Type
	Value    Value
}

func (c *Constant) Pos() token.Pos {
	return c.ConstPos
}

// A block represents a definition block in which a name may not be
// defined more than once.
type block struct {
	// The block enclosing this one, including blocks in other
	// scopes.
	outer *block
	// The nested block currently being compiled, or nil.
	inner *block
	// The Scope containing this block.
	scope *Scope
	// The Variables, Constants, and Types defined in this block.
	defs map[string]Def
	// The index of the first variable defined in this block.
	// This must be greater than the index of any variable defined
	// in any parent of this block within the same Scope at the
	// time this block is entered.
	offset int
	// The number of Variables defined in this block.
	numVars int
	// If global, do not allocate new vars and consts in
	// the frame; assume that the refs will be compiled in
	// using defs[name].Init.
	global bool
}

// A Scope is the compile-time analogue of a Frame, which captures
// some subtree of blocks.
type Scope struct {
	// The root block of this scope.
	*block
	// The maximum number of variables required at any point in
	// this Scope.  This determines the number of slots needed in
	// Frame's created from this Scope at run-time.
	maxVars int
}

func (b *block) enterChild() *block {
	if b.inner != nil && b.inner.scope == b.scope {
		log.Panic("Failed to exit child block before entering another child")
	}
	sub := &block{
		outer:  b,
		scope:  b.scope,
		defs:   make(map[string]Def),
		offset: b.offset + b.numVars,
	}
	b.inner = sub
	return sub
}

func (b *block) exit() {
	if b.outer == nil {
		log.Panic("Cannot exit top-level block")
	}
	if b.outer.scope == b.scope {
		if b.outer.inner != b {
			log.Panic("Already exited block")
		}
		if b.inner != nil && b.inner.scope == b.scope {
			log.Panic("Exit of parent block without exit of child block")
		}
	}
	b.outer.inner = nil
}

func (b *block) ChildScope() *Scope {
	if b.inner != nil && b.inner.scope == b.scope {
		log.Panic("Failed to exit child block before entering a child scope")
	}
	sub := b.enterChild()
	sub.offset = 0
	sub.scope = &Scope{sub, 0}
	return sub.scope
}

func (b *block) DefineVar(name string, pos token.Pos, t Type) (*Variable, Def) {
	if prev, ok := b.defs[name]; ok {
		return nil, prev
	}
	v := b.defineSlot(t, false)
	v.VarPos = pos
	b.defs[name] = v
	return v, nil
}

func (b *block) DefineTemp(t Type) *Variable { return b.defineSlot(t, true) }

func (b *block) defineSlot(t Type, temp bool) *Variable {
	if b.inner != nil && b.inner.scope == b.scope {
		log.Panic("Failed to exit child block before defining variable")
	}
	index := -1
	if !b.global || temp {
		index = b.offset + b.numVars
		b.numVars++
		if index >= b.scope.maxVars {
			b.scope.maxVars = index + 1
		}
	}
	v := &Variable{token.NoPos, index, t, nil}
	return v
}

func (b *block) DefineConst(name string, pos token.Pos, t Type, v Value) (*Constant, Def) {
	if prev, ok := b.defs[name]; ok {
		return nil, prev
	}
	c := &Constant{pos, t, v}
	b.defs[name] = c
	return c, nil
}

func (b *block) DefineType(name string, pos token.Pos, t Type) Type {
	if _, ok := b.defs[name]; ok {
		return nil
	}
	nt := &NamedType{pos, name, nil, true, make(map[string]Method)}
	if t != nil {
		nt.Complete(t)
	}
	b.defs[name] = nt
	return nt
}

func (b *block) Lookup(name string) (bl *block, level int, def Def) {
	for b != nil {
		if d, ok := b.defs[name]; ok {
			return b, level, d
		}
		if b.outer != nil && b.scope != b.outer.scope {
			level++
		}
		b = b.outer
	}
	return nil, 0, nil
}

func (s *Scope) NewFrame(outer *Frame) *Frame { return outer.child(s.maxVars) }

/*
 * Frames
 */

type Frame struct {
	Outer *Frame
	Vars  []Value
}

func (f *Frame) Get(level int, index int) Value {
	for ; level > 0; level-- {
		f = f.Outer
	}
	return f.Vars[index]
}

func (f *Frame) child(numVars int) *Frame {
	// TODO(austin) This is probably rather expensive.  All values
	// require heap allocation and zeroing them when we execute a
	// definition typically requires some computation.
	return &Frame{f, make([]Value, numVars)}
}
