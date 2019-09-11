// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"encoding/json"
	"fmt"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
)

func (s *Server) hover(ctx context.Context, params *protocol.HoverParams) (*protocol.Hover, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.session.ViewOf(uri)
	f, err := getGoFile(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	ident, err := source.Identifier(ctx, view, f, params.Position)
	if err != nil {
		return nil, nil
	}
	hover, err := ident.Hover(ctx)
	if err != nil {
		return nil, err
	}
	rng, err := ident.Range()
	if err != nil {
		return nil, err
	}
	contents := s.toProtocolHoverContents(ctx, hover, view.Options())
	return &protocol.Hover{
		Contents: contents,
		Range:    &rng,
	}, nil
}

func (s *Server) toProtocolHoverContents(ctx context.Context, h *source.HoverInformation, options source.Options) protocol.MarkupContent {
	content := protocol.MarkupContent{
		Kind: options.PreferredContentFormat,
	}
	signature := h.Signature
	if content.Kind == protocol.Markdown {
		signature = fmt.Sprintf("```go\n%s\n```", h.Signature)
	}
	switch options.HoverKind {
	case source.SingleLine:
		content.Value = h.SingleLine
	case source.NoDocumentation:
		content.Value = signature
	case source.SynopsisDocumentation:
		if h.Synopsis != "" {
			content.Value = fmt.Sprintf("%s\n%s", h.Synopsis, signature)
		} else {
			content.Value = signature
		}
	case source.FullDocumentation:
		if h.FullDocumentation != "" {
			content.Value = fmt.Sprintf("%s\n%s", signature, h.FullDocumentation)
		} else {
			content.Value = signature
		}
	case source.Structured:
		b, err := json.Marshal(h)
		if err != nil {
			log.Error(ctx, "failed to marshal structured hover", err)
		}
		content.Value = string(b)
	}
	return content
}
