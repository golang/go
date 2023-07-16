// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

// The inline command applies the inliner to the specified packages of
// Go source code. Run with:
//
//	$ go run ./internal/refactor/inline/analyzer/main.go -fix packages...
package main

import (
	"golang.org/x/tools/go/analysis/singlechecker"
	inlineanalyzer "golang.org/x/tools/internal/refactor/inline/analyzer"
)

func main() { singlechecker.Main(inlineanalyzer.Analyzer) }
