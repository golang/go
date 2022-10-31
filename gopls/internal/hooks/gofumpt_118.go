// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package hooks

import (
	"context"

	"golang.org/x/tools/gopls/internal/lsp/source"
	"mvdan.cc/gofumpt/format"
)

func updateGofumpt(options *source.Options) {
	options.GofumptFormat = func(ctx context.Context, langVersion, modulePath string, src []byte) ([]byte, error) {
		return format.Source(src, format.Options{
			LangVersion: langVersion,
			ModulePath:  modulePath,
		})
	}
}
