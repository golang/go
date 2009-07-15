// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"eval";
)

func NewRootScope() *Scope {
	return &Scope{nil, make(map[string] Def), 0};
}

func (s *Scope) Fork() *Scope {
	return &Scope{s, make(map[string] Def), 0};
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

func (s *Scope) DefineConst(name string, v Value) *Constant {
	if _, ok := s.defs[name]; ok {
		return nil;
	}
	c := &Constant{v.Type(), v};
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
