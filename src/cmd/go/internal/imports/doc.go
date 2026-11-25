// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package imports scans Go source files for their import
// declarations, evaluating build constraints, without type-checking
// or otherwise loading packages. The module loader uses it to discover
// the import graph, and the package loader and module index use its
// evaluation of build constraints.
//
// [ScanDir] and [ScanFiles] return the imports and test imports of
// the files of a directory or list that match a set of build tags.
// [ReadImports] and [ReadComments] read only the prefix of a file
// that holds its imports or its leading comments.
//
// [ShouldBuild], [MatchFile], and [Eval] evaluate build constraints
// (//go:build lines, and file name suffixes such as _linux.go)
// against a set of tags. [Tags] returns the tags of the target
// platform; [AnyTags] returns a set that satisfies nearly every
// constraint, for commands such as go mod tidy and go mod vendor that
// must consider all platforms.
package imports
