// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package analysis defines the core types of the Go analysis
// framework, a modular interface between static checkers
// ("analyzers") and the drivers that run them.
//
// An [Analyzer] statically describes an analysis function: its name,
// documentation, prerequisite analyzers, and the types of its result
// and facts. A [Pass] provides an analyzer with information about a
// single package and with operations for reporting [Diagnostic]
// values and for importing and exporting [Fact] values.
//
// This package defines only the types. The analyzers, drivers, and
// supporting libraries are in golang.org/x/tools/go/analysis and its
// subpackages; the types of golang.org/x/tools/go/analysis itself are
// aliases for the ones defined here. See that package for full
// documentation.
package analysis
