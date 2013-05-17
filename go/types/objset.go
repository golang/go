// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"bytes"
	"fmt"
	"go/ast"
)

// An ObjSet maintains a set of objects identified by
// their name and package that declares them.
//
type ObjSet struct {
	entries []Object // set entries in insertion order
}

// Lookup returns the object with the given package and name
// if it is found in ObjSet s, otherwise it returns nil.
//
func (s *ObjSet) Lookup(pkg *Package, name string) Object {
	for _, obj := range s.entries {
		// spec:
		// "Two identifiers are different if they are spelled differently,
		// or if they appear in different packages and are not exported.
		// Otherwise, they are the same."
		if obj.Name() == name && (ast.IsExported(name) || obj.Pkg().path == pkg.path) {
			return obj
		}
	}
	return nil
}

// Insert attempts to insert an object obj into ObjSet s.
// If s already contains an object from the same package
// with the same name, Insert leaves s unchanged and returns
// that object. Otherwise it inserts obj and returns nil.
//
func (s *ObjSet) Insert(obj Object) Object {
	pkg := obj.Pkg()
	name := obj.Name()
	assert(obj.Type() != nil)
	if alt := s.Lookup(pkg, name); alt != nil {
		return alt
	}
	s.entries = append(s.entries, obj)
	return nil
}

// Debugging support
func (s *ObjSet) String() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "ObjSet %p {", s)
	if s != nil && len(s.entries) > 0 {
		fmt.Fprintln(&buf)
		for _, obj := range s.entries {
			fmt.Fprintf(&buf, "\t%s.%s\t%T\n", obj.Pkg().path, obj.Name(), obj)
		}
	}
	fmt.Fprintf(&buf, "}\n")
	return buf.String()
}
