// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"eval";
)

func NewRootScope() *Scope {
	return &Scope{defs: make(map[string] Def)};
}

func (s *Scope) Fork() *Scope {
	return &Scope{outer: s, defs: make(map[string] Def)};
}

func (s *Scope) DefineVar(name string, t Type) *Variable {
	if _, ok := s.defs[name]; ok {
		return nil;
	}
	v := &Variable{s.numVars, t};
	s.numVars++;
	s.defs[name] = v;
	return v;
}

func (s *Scope) DefineConst(name string, t Type, v Value) *Constant {
	if _, ok := s.defs[name]; ok {
		return nil;
	}
	c := &Constant{t, v};
	s.defs[name] = c;
	return c;
}

func (s *Scope) DefineType(name string, t Type) bool {
	if _, ok := s.defs[name]; ok {
		return false;
	}
	s.defs[name] = t;
	return true;
}

func (s *Scope) Lookup(name string) (Def, *Scope) {
	for s != nil {
		if d, ok := s.defs[name]; ok {
			return d, s;
		}
		s = s.outer;
	}
	return nil, nil;
}

func (s *Scope) NewFrame(outer *Frame) *Frame {
	if s.varTypes == nil {
		// First creation of a frame from this scope.  Compute
		// and memoize the types of all variables.
		ts := make([]Type, s.numVars);
		for _, d := range s.defs {
			if v, ok := d.(*Variable); ok {
				ts[v.Index] = v.Type;
			}
		}
		s.varTypes = ts;
	}

	// Create frame
	vars := make([]Value, s.numVars);
	for i, t := range s.varTypes {
		vars[i] = t.Zero();
	}
	return &Frame{outer, s, vars};
}

func (f *Frame) Get(s *Scope, index int) Value {
	for f.Scope != s {
		f = f.Outer;
	}
	return f.Vars[index];
}
