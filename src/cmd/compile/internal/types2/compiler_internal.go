// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"fmt"
)

// This file should not be copied to go/types.  See go.dev/issue/67477

// RenameResult takes an array of (result) fields and an index, and if the indexed field
// does not have a name and if the result in the signature also does not have a name,
// then the signature and field are renamed to
//
//	fmt.Sprintf("#rv%d", i+1)
//
// the newly named object is inserted into the signature's scope,
// and the object and new field name are returned.
//
// The intended use for RenameResult is to allow rangefunc to assign results within a closure.
// This is a hack, as narrowly targeted as possible to discourage abuse.
func (s *Signature) RenameResult(results []*syntax.Field, i int) (*Var, *syntax.Name) {
	a := results[i]
	obj := s.Results().At(i)

	if !(obj.name == "" || obj.name == "_" && a.Name == nil || a.Name.Value == "_") {
		panic("Cannot change an existing name")
	}

	pos := a.Pos()
	typ := a.Type.GetTypeInfo().Type

	name := fmt.Sprintf("#rv%d", i+1)
	obj.name = name
	s.scope.Insert(obj)
	obj.setScopePos(pos)

	tv := syntax.TypeAndValue{Type: typ}
	tv.SetIsValue()

	n := syntax.NewName(pos, obj.Name())
	n.SetTypeInfo(tv)

	a.Name = n

	return obj, n
}

// Comment returns the scope comment, for debugging purposes.
func (s *Scope) Comment() string { return s.comment }

// LookupParent follows the parent chain of scopes starting with s until
// it finds an object named name, considering only objects whose scope
// begins at or before pos. It returns nil when there is no such object.
// This mirrors go/types (*Scope).LookupParent.
func (s *Scope) LookupParent(name string, pos syntax.Pos) Object {
	for ; s != nil; s = s.parent {
		if obj := s.Lookup(name); obj != nil {
			if sp := obj.scopePos(); !pos.IsKnown() || !sp.IsKnown() || sp.Cmp(pos) <= 0 {
				return obj
			}
		}
	}
	return nil
}

// Innermost returns the innermost child scope containing pos, or nil when
// pos is outside every child scope. This mirrors go/types (*Scope).Innermost.
func (s *Scope) Innermost(pos syntax.Pos) *Scope {
	if s.parent == Universe {
		for _, child := range s.children {
			if inner := child.Innermost(pos); inner != nil {
				return inner
			}
		}
	}
	if s.contains(pos) {
		for _, child := range s.children {
			if child.contains(pos) {
				return child.Innermost(pos)
			}
		}
		return s
	}
	return nil
}

// contains reports whether pos lies in the scope's extent [pos, end).
func (s *Scope) contains(pos syntax.Pos) bool {
	return s.pos.IsKnown() && pos.IsKnown() && s.pos.Cmp(pos) <= 0 && pos.Cmp(s.end) < 0
}
