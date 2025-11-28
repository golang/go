// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package refactor provides operators to compute common textual edits
// for refactoring tools.
//
// This package should not use features of the analysis API
// other than [analysis.TextEdit].
package refactor

import (
	"fmt"
	"go/token"
	"go/types"
)

// FreshName returns the name of an identifier that is undefined
// at the specified position, based on the preferred name.
//
// TODO(adonovan): refine this to choose a fresh name only when there
// would be a conflict with the existing declaration: it's fine to
// redeclare a name in a narrower scope so long as there are no free
// references to the outer name from within the narrower scope.
func FreshName(scope *types.Scope, pos token.Pos, preferred string) string {
	newName := preferred
	for i := 0; ; i++ {
		if _, obj := scope.LookupParent(newName, pos); obj == nil {
			break // fresh
		}
		newName = fmt.Sprintf("%s%d", preferred, i)
	}
	return newName
}
