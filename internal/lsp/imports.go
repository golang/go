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
	f, err := v.GetFile(ctx, fromProtocolURI(uri))
	if err != nil {
		return nil, err
	}
	tok, err := f.GetToken()
	if err != nil {
		return nil, err
	}
	r := source.Range{
		Start: tok.Pos(0),
		End:   tok.Pos(tok.Size()),
	}
	content, err := f.Read()
	if err != nil {
		return nil, err
	}
	edits, err := source.Imports(ctx, tok.Name(), content, r)
	if err != nil {
		return nil, err
	}
	return toProtocolEdits(tok, content, edits), nil
}
