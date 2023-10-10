// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The httpmux command runs the httpmux analyzer.
package main

import (
	"golang.org/x/tools/go/analysis/passes/httpmux"
	"golang.org/x/tools/go/analysis/singlechecker"
)

func main() { singlechecker.Main(httpmux.Analyzer) }
