// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gotestchecker

// This file declares back doors for tests.

import (
	"go/analysis"
	"go/types"
)

// Analyzers returns the valid analyzers reachable from vars, in the
// canonical order that identifies them in the protocol.
func Analyzers(vars []Var) []*analysis.Analyzer {
	return newGraph(vars).analyzers
}

// A FactSet is the set of facts visible to the analysis of a package.
type FactSet = factSet

// RegisterFactType registers a fact type for encoding.
func RegisterFactType(f analysis.Fact) { registerFactType(f) }

// DecodeFacts returns the set of facts for package pkg, decoding the
// facts of its direct imports, as returned by read.
func DecodeFacts(pkg *types.Package, getPackage func(pkgPath string) *types.Package, read func(pkgPath string) ([]byte, error)) (*FactSet, error) {
	set, _, _, err := newFactDecoder(pkg, getPackage).Decode(read)
	return set, err
}
