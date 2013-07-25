// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements objsets.
//
// An objset is similar to a Scope but objset elements
// are identified by their unique id, instead of their
// object name.

package types

// An objset is a set of objects identified by their unique id.
// The zero value for objset is a ready-to-use empty objset.
type objset struct {
	elems map[string]Object // allocated lazily
}

// insert attempts to insert an object obj into objset s.
// If s already contains an alternative object alt with
// the same name, insert leaves s unchanged and returns alt.
// Otherwise it inserts obj and returns nil. Objects with
// blank "_" names are ignored.
func (s *objset) insert(obj Object) Object {
	name := obj.Name()
	if name == "_" {
		return nil
	}
	id := Id(obj.Pkg(), name)
	if alt := s.elems[id]; alt != nil {
		return alt
	}
	if s.elems == nil {
		s.elems = make(map[string]Object)
	}
	s.elems[id] = obj
	return nil
}
