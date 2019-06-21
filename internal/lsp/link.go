// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"strconv"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
)

func (s *Server) documentLink(ctx context.Context, params *protocol.DocumentLinkParams) ([]protocol.DocumentLink, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.session.ViewOf(uri)
	f, m, err := getGoFile(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	file := f.GetAST(ctx)
	if file == nil {
		return nil, fmt.Errorf("no AST for %v", uri)
	}
	// Add a Godoc link for each imported package.
	var result []protocol.DocumentLink
	for _, imp := range file.Imports {
		spn, err := span.NewRange(view.Session().Cache().FileSet(), imp.Pos(), imp.End()).Span()
		if err != nil {
			return nil, err
		}
		rng, err := m.Range(spn)
		if err != nil {
			return nil, err
		}
		target, err := strconv.Unquote(imp.Path.Value)
		if err != nil {
			continue
		}
		target = "https://godoc.org/" + target
		result = append(result, protocol.DocumentLink{
			Range:  rng,
			Target: target,
		})
	}
	return result, nil
}
