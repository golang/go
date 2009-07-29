// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"eval";
	"fmt";
	"log";
)

func (b *block) enterChild() *block {
	if b.inner != nil {
		log.Crash("Failed to exit child block before entering another child");
	}
	sub := &block{
		outer: b,
		scope: b.scope,
		defs: make(map[string] Def),
		offset: b.offset+b.numVars,
	};
	b.inner = sub;
	return sub;
}

func (b *block) exit() {
	if b.outer == nil {
		log.Crash("Cannot exit top-level block");
	}
	if b.outer.inner != b {
		log.Crash("Already exited block");
	}
	if b.inner != nil {
		log.Crash("Exit of parent block without exit of child block");
	}
	b.outer.inner = nil;
}

func (b *block) ChildScope() *Scope {
	if b.inner != nil {
		log.Crash("Failed to exit child block before entering a child scope");
	}
	sub := b.enterChild();
	sub.offset = 0;
	sub.scope = &Scope{sub, 0};
	return sub.scope;
}

func (b *block) DefineVar(name string, t Type) *Variable {
	if _, ok := b.defs[name]; ok {
		return nil;
	}
	v := b.DefineTemp(t);
	if v != nil {
		b.defs[name] = v;
	}
	return v;
}

func (b *block) DefineTemp(t Type) *Variable {
	if b.inner != nil {
		log.Crash("Failed to exit child block before defining variable");
	}
	index := b.offset+b.numVars;
	v := &Variable{index, t};
	b.numVars++;
	if index+1 > b.scope.maxVars {
		b.scope.maxVars = index+1;
	}
	return v;
}

func (b *block) DefineConst(name string, t Type, v Value) *Constant {
	if _, ok := b.defs[name]; ok {
		return nil;
	}
	c := &Constant{t, v};
	b.defs[name] = c;
	return c;
}

func (b *block) DefineType(name string, t Type) Type {
	if _, ok := b.defs[name]; ok {
		return nil;
	}
	// We take the representative type of t because multiple
	// levels of naming are useless.
	nt := &NamedType{name, t.rep()};
	b.defs[name] = nt;
	return nt;
}

func (b *block) Lookup(name string) (level int, def Def) {
	for b != nil {
		if d, ok := b.defs[name]; ok {
			return level, d;
		}
		if b.outer != nil && b.scope != b.outer.scope {
			level++;
		}
		b = b.outer;
	}
	return 0, nil;
}

func (s *Scope) NewFrame(outer *Frame) *Frame {
	return outer.child(s.maxVars);
}

func (f *Frame) Get(level int, index int) Value {
	for ; level > 0; level-- {
		f = f.Outer;
	}
	return f.Vars[index];
}

func (f *Frame) child(numVars int) *Frame {
	// TODO(austin) This is probably rather expensive.  All values
	// require heap allocation and zeroing them when we execute a
	// definition typically requires some computation.
	return &Frame{f, make([]Value, numVars)};
}
