// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package hooks adds all the standard gopls implementations.
// This can be used in tests without needing to use the gopls main, and is
// also the place to edit for custom builds of gopls.
package hooks // import "golang.org/x/tools/gopls/internal/hooks"

import (
	"golang.org/x/tools/internal/lsp/source"
)

func Options(options *source.Options) {
	if options.GoDiff {
		options.ComputeEdits = ComputeEdits
	}
	updateAnalyzers(options)
}
