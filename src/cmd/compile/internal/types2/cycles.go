// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "cmd/compile/internal/syntax"

// directCycles searches for direct cycles among package level type declarations.
// See directCycle for details.
func (check *Checker) directCycles() {
	pathIdx := make(map[*TypeName]int)
	for _, obj := range check.objList {
		if tname, ok := obj.(*TypeName); ok {
			check.directCycle(tname, pathIdx)
		}
	}
}

// directCycle checks if the declaration of the type given by tname contains a direct cycle.
// A direct cycle exists if the path from tname's declaration's RHS leads from type name to
// type name and eventually ends up on that path again, via regular or alias declarations;
// in other words if there are no type literals (or basic types) on the path, and the path
// doesn't end in an undeclared object.
// If a cycle is detected, a cycle error is reported and the type at the start of the cycle
// is marked as invalid.
//
// The pathIdx map tracks which type names have been processed. An entry can be
// in 1 of 3 states as used in a typical 3-state (white/grey/black) graph marking
// algorithm for cycle detection:
//
//   - entry not found: tname has not been seen before (white)
//   - value is >= 0  : tname has been seen but is not done (grey); the value is the path index
//   - value is <  0  : tname has been seen and is done (black)
//
// When directCycle returns, the pathIdx entries for all type names on the path
// that starts at tname are marked black, regardless of whether there was a cycle.
// This ensures that a type name is traversed only once.
func (check *Checker) directCycle(tname *TypeName, pathIdx map[*TypeName]int) {
	if debug && check.conf.Trace {
		check.trace(tname.Pos(), "-- check direct cycle for %s", tname)
	}

	var path []*TypeName
	for {
		start, found := pathIdx[tname]
		if start < 0 {
			// tname is marked black - do not traverse it again.
			// (start can only be < 0 if it was found in the first place)
			break
		}

		if found {
			// tname is marked grey - we have a cycle on the path beginning at start.
			// Mark tname as invalid.
			tname.setType(Typ[Invalid])

			// collect type names on cycle
			var cycle []Object
			for _, tname := range path[start:] {
				cycle = append(cycle, tname)
			}

			check.cycleError(cycle, firstInSrc(cycle))
			break
		}

		// tname is marked white - mark it grey and add it to the path.
		pathIdx[tname] = len(path)
		path = append(path, tname)

		// For direct cycle detection, we don't care about whether we have an alias or not.
		// If the associated type is not a name, we're at the end of the path and we're done.
		rhs, ok := check.objMap[tname].tdecl.Type.(*syntax.Name)
		if !ok {
			break
		}

		// Determine the RHS type. If it is not found in the package scope, we either
		// have an error (which will be reported later), or the type exists elsewhere
		// (universe scope, file scope via dot-import) and a cycle is not possible in
		// the first place. If it is not a type name, we cannot have a direct cycle
		// either. In all these cases we can stop.
		tname1, ok := check.pkg.scope.Lookup(rhs.Value).(*TypeName)
		if !ok {
			break
		}

		// Otherwise, continue with the RHS.
		tname = tname1
	}

	// Mark all traversed type names black.
	// (ensure that pathIdx doesn't contain any grey entries upon returning)
	for _, tname := range path {
		pathIdx[tname] = -1
	}

	if debug {
		for _, i := range pathIdx {
			assert(i < 0)
		}
	}
}
