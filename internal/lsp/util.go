// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

func getGoFile(ctx context.Context, view source.View, uri span.URI) (source.GoFile, error) {
	f, err := view.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	gof, ok := f.(source.GoFile)
	if !ok {
		return nil, errors.Errorf("%s is not a Go file", uri)
	}
	return gof, nil
}
