// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package hooks adds all the standard gopls implementations.
// This can be used in tests without needing to use the gopls main, and is
// also the place to edit for custom builds of gopls.
package hooks // import "golang.org/x/tools/gopls/internal/hooks"

import (
	"context"
	"regexp"

	"golang.org/x/tools/internal/lsp/source"
	"mvdan.cc/gofumpt/format"
	"mvdan.cc/xurls/v2"
)

func Options(options *source.Options) {
	options.LicensesText = licensesText
	if options.GoDiff {
		options.ComputeEdits = ComputeEdits
	}
	options.URLRegexp = relaxedFullWord
	options.GofumptFormat = func(ctx context.Context, src []byte) ([]byte, error) {
		return format.Source(src, format.Options{})
	}
	updateAnalyzers(options)
}

var relaxedFullWord *regexp.Regexp

// Ensure links are matched as full words, not anywhere.
func init() {
	relaxedFullWord = regexp.MustCompile(`\b(` + xurls.Relaxed().String() + `)\b`)
	relaxedFullWord.Longest()
}
