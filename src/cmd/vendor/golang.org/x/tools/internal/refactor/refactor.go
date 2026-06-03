// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package refactor provides operators to compute common textual edits
// for refactoring tools.
//
// This package should not use features of the analysis API other than [Edit].
package refactor

import (
	"fmt"
	"go/token"
	"go/types"
)

// FreshName returns the name of an identifier that is undefined
// at the specified position, based on the preferred name.
//
// export/use freshName in go/analysis/passes/modernize/modernize.go if you want
// to generate a fresh name only when necessary (i.e., there is both an existing
// declaration and some free reference to the name within a narrower scope)
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
