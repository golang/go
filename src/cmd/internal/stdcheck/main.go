// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/analysis/fipsdit"
	"internal/analysis/unitchecker"
)

func main() {
	unitchecker.Main(fipsdit.Analyzer)
}
