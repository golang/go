// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func organizeImports(ctx context.Context, v source.View, uri protocol.DocumentURI) ([]protocol.TextEdit, error) {
	sourceURI, err := fromProtocolURI(uri)
	if err != nil {
		return nil, err
	}
	f, err := v.GetFile(ctx, sourceURI)
	if err != nil {
		return nil, err
	}
	tok := f.GetToken()
	r := source.Range{
		Start: tok.Pos(0),
		End:   tok.Pos(tok.Size()),
	}
	edits, err := source.Imports(ctx, f, r)
	if err != nil {
		return nil, err
	}
	return toProtocolEdits(f, edits), nil
}
