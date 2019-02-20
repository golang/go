// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func organizeImports(ctx context.Context, v source.View, s span.Span) ([]protocol.TextEdit, error) {
	f, m, err := newColumnMap(ctx, v, s.URI)
	if err != nil {
		return nil, err
	}
	rng := s.Range(m.Converter)
	if rng.Start == rng.End {
		// if we have a single point, then assume the rest of the file
		rng.End = f.GetToken(ctx).Pos(f.GetToken(ctx).Size())
	}
	edits, err := source.Imports(ctx, f, rng)
	if err != nil {
		return nil, err
	}
	return toProtocolEdits(m, edits), nil
}
