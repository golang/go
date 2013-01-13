// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// A Scope maintains the set of named language entities declared
// in the scope and a link to the immediately surrounding (outer)
// scope.
//
type Scope struct {
	Outer   *Scope
	Entries []Object          // scope entries in insertion order
	large   map[string]Object // for fast lookup - only used for larger scopes
}

// Lookup returns the object with the given name if it is
// found in scope s, otherwise it returns nil. Outer scopes
// are ignored.
//
func (s *Scope) Lookup(name string) Object {
	if s.large != nil {
		return s.large[name]
	}
	for _, obj := range s.Entries {
		if obj.GetName() == name {
			return obj
		}
	}
	return nil
}

// Insert attempts to insert an object obj into scope s.
// If s already contains an object with the same name,
// Insert leaves s unchanged and returns that object.
// Otherwise it inserts obj and returns nil.
//
func (s *Scope) Insert(obj Object) Object {
	name := obj.GetName()
	if alt := s.Lookup(name); alt != nil {
		return alt
	}
	s.Entries = append(s.Entries, obj)

	// If the scope size reaches a threshold, use a map for faster lookups.
	const threshold = 20
	if len(s.Entries) > threshold {
		if s.large == nil {
			m := make(map[string]Object, len(s.Entries))
			for _, obj := range s.Entries {
				m[obj.GetName()] = obj
			}
			s.large = m
		}
		s.large[name] = obj
	}

	return nil
}
