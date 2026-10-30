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
	return freshName([]*types.Scope{scope}, []token.Pos{pos}, preferred)
}

// freshName returns an identifier derived from preferred that is
// undefined at all specified positions in their respective scopes.
//
// The scopes and poses slices must have the same length.
func freshName(scopes []*types.Scope, poses []token.Pos, preferred string) string {
	newName := preferred
outer:
	for i := 0; ; i++ {
		if i > 0 {
			newName = fmt.Sprintf("%s%d", preferred, i-1)
		}
		for j, pos := range poses {
			if s := scopes[j]; s != nil {
				if _, obj := s.LookupParent(newName, pos); obj != nil {
					continue outer
				}
			}
		}
		return newName
	}
}
