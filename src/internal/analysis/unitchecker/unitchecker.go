// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package unitchecker provides the unitchecker analysis driver, from
// the copy of golang.org/x/tools vendored into the standard library,
// to cmd/vet. Analyzers run by it must come from the same copy (see
// [internal/analysis/suite/vet]), since the *analysis.Analyzer types
// must be identical.
package unitchecker

import (
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/unitchecker"
)

// Main is the main function of a vet tool that runs the specified
// analyzers, as invoked by the go command. It never returns.
func Main(analyzers ...*analysis.Analyzer) {
	unitchecker.Main(analyzers...)
}
