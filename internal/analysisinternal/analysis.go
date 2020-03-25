// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package analysisinternal exposes internal-only fields from go/analysis.
package analysisinternal

import (
	"bytes"
	"go/token"
	"go/types"
)

func TypeErrorEndPos(fset *token.FileSet, src []byte, start token.Pos) token.Pos {
	// Get the end position for the type error.
	offset, end := fset.PositionFor(start, false).Offset, start
	if offset >= len(src) {
		return end
	}
	if width := bytes.IndexAny(src[offset:], " \n,():;[]+-*"); width > 0 {
		end = start + token.Pos(width)
	}
	return end
}

var GetTypeErrors = func(p interface{}) []types.Error { return nil }
var SetTypeErrors = func(p interface{}, errors []types.Error) {}

type TypeErrorPass string

const (
	NoNewVars      TypeErrorPass = "nonewvars"
	NoResultValues TypeErrorPass = "noresultvalues"
	UndeclaredName TypeErrorPass = "undeclaredname"
)
