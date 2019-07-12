// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func getSourceFile(ctx context.Context, v source.View, uri span.URI) (source.File, *protocol.ColumnMapper, error) {
	f, err := v.GetFile(ctx, uri)
	if err != nil {
		return nil, nil, err
	}
	data, _, err := f.Handle(ctx).Read(ctx)
	if err != nil {
		return nil, nil, err
	}
	tok, err := f.GetToken(ctx)
	if err != nil {
		return nil, nil, err
	}
	m := protocol.NewColumnMapper(f.URI(), f.URI().Filename(), f.FileSet(), tok, data)

	return f, m, nil
}

func getGoFile(ctx context.Context, v source.View, uri span.URI) (source.GoFile, *protocol.ColumnMapper, error) {
	f, m, err := getSourceFile(ctx, v, uri)
	if err != nil {
		return nil, nil, err
	}
	gof, ok := f.(source.GoFile)
	if !ok {
		return nil, nil, fmt.Errorf("not a Go file %v", f.URI())
	}
	return gof, m, nil
}
