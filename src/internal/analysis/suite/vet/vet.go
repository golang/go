// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package vet provides the analyzers of cmd/vet, from the copy of
// golang.org/x/tools vendored into the standard library, so that
// cmd/vet (and, in future, other parts of the toolchain, such as the
// analysis driver of go test) can share a single vendored copy.
//
// Run them with the driver from the same copy (see
// [internal/analysis/unitchecker]), since the *analysis.Analyzer types
// must be identical.
//
// This package is an import bottleneck: importing it for any symbol
// depends on every analyzer of the vet suite. That is intended, since
// it is meant only for programs that run the whole suite, such as
// cmd/vet; it should not be imported for the sake of a single
// analyzer.
package vet

import "golang.org/x/tools/go/analysis/suite/vet"

// Suite is the set of analyzers run by cmd/vet.
var Suite = vet.Suite
