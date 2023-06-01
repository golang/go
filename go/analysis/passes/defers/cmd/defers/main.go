// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The defers command runs the defers analyzer.
package main

import (
	"golang.org/x/tools/go/analysis/passes/defers"
	"golang.org/x/tools/go/analysis/singlechecker"
)

func main() { singlechecker.Main(defers.Analyzer) }
