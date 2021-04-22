// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package typeparams provides functions to work with type parameter data
// stored in the AST, while these AST changes are guarded by a build
// constraint.
package typeparams

// DisallowParsing is the numeric value of a parsing mode that disallows type
// parameters. This only matters if the typeparams experiment is active, and
// may be used for running tests that disallow generics.
const DisallowParsing = 1 << 30
