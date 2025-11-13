// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package goplsexport provides various backdoors to not-yet-published
// parts of x/tools that are needed by gopls.
package goplsexport

import "golang.org/x/tools/go/analysis"

var (
	ErrorsAsTypeModernizer *analysis.Analyzer // = modernize.errorsastypeAnalyzer
	StdIteratorsModernizer *analysis.Analyzer // = modernize.stditeratorsAnalyzer
	PlusBuildModernizer    *analysis.Analyzer // = modernize.plusbuildAnalyzer
)
