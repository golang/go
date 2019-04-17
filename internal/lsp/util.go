// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"strings"

	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// findView returns the view corresponding to the given URI.
// If the file is not already associated with a view, pick one using some heuristics.
func (s *Server) findView(ctx context.Context, uri span.URI) *cache.View {
	// first see if a view already has this file
	for _, view := range s.views {
		if view.FindFile(ctx, uri) != nil {
			return view
		}
	}
	var longest *cache.View
	for _, view := range s.views {
		if longest != nil && len(longest.Folder) > len(view.Folder) {
			continue
		}
		if strings.HasPrefix(string(uri), string(view.Folder)) {
			longest = view
		}
	}
	if longest != nil {
		return longest
	}
	//TODO: are there any more heuristics we can use?
	return s.views[0]
}

func newColumnMap(ctx context.Context, v source.View, uri span.URI) (source.File, *protocol.ColumnMapper, error) {
	f, err := v.GetFile(ctx, uri)
	if err != nil {
		return nil, nil, err
	}
	tok := f.GetToken(ctx)
	if tok == nil {
		return nil, nil, fmt.Errorf("no file information for %v", f.URI())
	}
	m := protocol.NewColumnMapper(f.URI(), f.GetFileSet(ctx), tok, f.GetContent(ctx))
	return f, m, nil
}
