// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19
// +build go1.19

package main

// assign names to types. many types come with names, but names
// have to be provided for "or", "and", "tuple", and "literal" types.
// Only one tuple type occurs, so it poses no problem. Otherwise
// the name cannot depend on the ordering of the components, as permuting
// them doesn't change the type. One possibility is to build the name
// of the type out of the names of its components, done in an
// earlier version of this code, but rejected by code reviewers.
// (the name would change if the components changed.)
// An alternate is to use the definition context, which is what is done here
// and works for the existing code. However, it cannot work in general.
// (This easiest case is an "or" type with two "literal" components.
// The components will get the same name, as their definition contexts
// are identical.) spec.byName contains enough information to detect
// such cases. (Note that sometimes giving the same name to different
// types is correct, for instance when they involve stringLiterals.)

import (
	"strings"
)

// stacks contain information about the ancestry of a type
// (spaces and initial capital letters are treated specially in stack.name())
type stack []string

func (s stack) push(v string) stack {
	return append(s, v)
}

func (s stack) pop() {
	s = s[:len(s)-1]
}

// generate a type name from the stack that contains its ancestry
//
// For instance, ["Result textDocument/implementation"] becomes "_textDocument_implementation"
// which, after being returned, becomes "Or_textDocument_implementation",
// which will become "[]Location" eventually (for gopls compatibility).
func (s stack) name(prefix string) string {
	var nm string
	var seen int
	// use the most recent 2 entries, if there are 2,
	// or just the only one.
	for i := len(s) - 1; i >= 0 && seen < 2; i-- {
		x := s[i]
		if x[0] <= 'Z' && x[0] >= 'A' {
			// it may contain a message
			if idx := strings.Index(x, " "); idx >= 0 {
				x = prefix + strings.Replace(x[idx+1:], "/", "_", -1)
			}
			nm += x
			seen++
		}
	}
	return nm
}
