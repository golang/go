// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"go/types"
	"strings"
)

// deepCompletionState stores our state as we search for deep completions.
// "deep completion" refers to searching into objects' fields and methods to
// find more completion candidates.
type deepCompletionState struct {
	// enabled is true if deep completions are enabled.
	enabled bool

	// chain holds the traversal path as we do a depth-first search through
	// objects' members looking for exact type matches.
	chain []types.Object

	// chainNames holds the names of the chain objects. This allows us to
	// save allocations as we build many deep completion items.
	chainNames []string
}

// push pushes obj onto our search stack.
func (s *deepCompletionState) push(obj types.Object) {
	s.chain = append(s.chain, obj)
	s.chainNames = append(s.chainNames, obj.Name())
}

// pop pops the last object off our search stack.
func (s *deepCompletionState) pop() {
	s.chain = s.chain[:len(s.chain)-1]
	s.chainNames = s.chainNames[:len(s.chainNames)-1]
}

// chainString joins the chain of objects' names together on ".".
func (s *deepCompletionState) chainString(finalName string) string {
	s.chainNames = append(s.chainNames, finalName)
	chainStr := strings.Join(s.chainNames, ".")
	s.chainNames = s.chainNames[:len(s.chainNames)-1]
	return chainStr
}

func (c *completer) inDeepCompletion() bool {
	return len(c.deepState.chain) > 0
}

// deepSearch searches through obj's subordinate objects for more
// completion items.
func (c *completer) deepSearch(obj types.Object) {
	if !c.deepState.enabled {
		return
	}

	// Don't search into type names.
	if isTypeName(obj) {
		return
	}

	// Don't search embedded fields because they were already included in their
	// parent's fields.
	if v, ok := obj.(*types.Var); ok && v.Embedded() {
		return
	}

	// Push this object onto our search stack.
	c.deepState.push(obj)

	switch obj := obj.(type) {
	case *types.PkgName:
		c.packageMembers(obj)
	default:
		// For now it is okay to assume obj is addressable since we don't search beyond
		// function calls.
		c.methodsAndFields(obj.Type(), true)
	}

	// Pop the object off our search stack.
	c.deepState.pop()
}
