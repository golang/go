// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"eval";
	"fmt";
)

func (s *Scope) Fork() *Scope {
	return &Scope{
		outer: s,
		defs: make(map[string] Def),
		temps: make(map[int] *Variable)
	};
}

func (s *Scope) DefineVar(name string, t Type) *Variable {
	if _, ok := s.defs[name]; ok {
		return nil;
	}
	v := &Variable{s.numVars, t};
	s.defs[name] = v;
	s.numVars++;
	return v;
}

func (s *Scope) DefineTemp(t Type) *Variable {
	v := &Variable{s.numVars, t};
	s.temps[s.numVars] = v;
	s.numVars++;
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

func (s *Scope) DefineType(name string, t Type) Type {
	if _, ok := s.defs[name]; ok {
		return nil;
	}
	// We take the representative type of t because multiple
	// levels of naming are useless.
	nt := &NamedType{s, name, t.rep()};
	s.defs[name] = nt;
	return nt;
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
				// Record the representative type to
				// avoid indirecting through named
				// types every time we drop a frame.
				ts[v.Index] = v.Type.rep();
			}
		}
		for _, v := range s.temps {
			ts[v.Index] = v.Type.rep();
		}
		s.varTypes = ts;
	}

	// Create frame
	//
	// TODO(austin) This is probably rather expensive.  All values
	// require heap allocation and the Zero method typically
	// requires some computation.
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

func stringFrame(f *Frame) (string, string) {
	res := "";
	indent := "";
	if f.Outer != nil {
		res, indent = stringFrame(f.Outer);
	}

	names := make([]string, f.Scope.numVars);
	types := make([]Type, f.Scope.numVars);
	for name, def := range f.Scope.defs {
		def, ok := def.(*Variable);
		if !ok {
			continue;
		}
		names[def.Index] = name;
		types[def.Index] = def.Type;
	}
	for _, def := range f.Scope.temps {
		names[def.Index] = "(temp)";
		types[def.Index] = def.Type;
	}

	for i, val := range f.Vars {
		res += fmt.Sprintf("%s%-10s %-10s %s\n", indent, names[i], types[i], val);
	}
	return res, indent + "  ";
}

func (f *Frame) String() string {
	res, _ := stringFrame(f);
	return res;
}
