// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: show that two-non-empty dotjoin can happen, by using an anon struct as a field type
// TODO: don't report removed/changed methods for both value and pointer method sets?

package apidiff

import (
	"fmt"
	"go/types"
	"sort"
	"strings"
)

// There can be at most one message for each object or part thereof.
// Parts include interface methods and struct fields.
//
// The part thing is necessary. Method (Func) objects have sufficient info, but field
// Vars do not: they just have a field name and a type, without the enclosing struct.
type messageSet map[types.Object]map[string]string

// Add a message for obj and part, overwriting a previous message
// (shouldn't happen).
// obj is required but part can be empty.
func (m messageSet) add(obj types.Object, part, msg string) {
	s := m[obj]
	if s == nil {
		s = map[string]string{}
		m[obj] = s
	}
	if f, ok := s[part]; ok && f != msg {
		fmt.Printf("! second, different message for obj %s, part %q\n", obj, part)
		fmt.Printf("  first:  %s\n", f)
		fmt.Printf("  second: %s\n", msg)
	}
	s[part] = msg
}

func (m messageSet) collect() []string {
	var s []string
	for obj, parts := range m {
		// Format each object name relative to its own package.
		objstring := objectString(obj)
		for part, msg := range parts {
			var p string

			if strings.HasPrefix(part, ",") {
				p = objstring + part
			} else {
				p = dotjoin(objstring, part)
			}
			s = append(s, p+": "+msg)
		}
	}
	sort.Strings(s)
	return s
}

func objectString(obj types.Object) string {
	if f, ok := obj.(*types.Func); ok {
		sig := f.Type().(*types.Signature)
		if recv := sig.Recv(); recv != nil {
			tn := types.TypeString(recv.Type(), types.RelativeTo(obj.Pkg()))
			if tn[0] == '*' {
				tn = "(" + tn + ")"
			}
			return fmt.Sprintf("%s.%s", tn, obj.Name())
		}
	}
	return obj.Name()
}

func dotjoin(s1, s2 string) string {
	if s1 == "" {
		return s2
	}
	if s2 == "" {
		return s1
	}
	return s1 + "." + s2
}
