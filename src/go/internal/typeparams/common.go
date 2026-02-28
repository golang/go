// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package typeparams provides functions to work with type parameter data
// stored in the AST, while these AST changes are guarded by a build
// constraint.
package typeparams

// 'Hidden' parser modes to control the parsing of type-parameter related
// features.
const (
	DisallowTypeSets = 1 << 29 // Disallow eliding 'interface' in constraint type sets.
	DisallowParsing  = 1 << 30 // Disallow type parameters entirely.
)
