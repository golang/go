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
	view, err := s.session.ViewOf(uri)
	if err != nil {
		return nil, err
	}
	snapshot := view.Snapshot()
	f, err := view.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	ident, err := source.Identifier(ctx, snapshot, f, params.Position)
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
		Range:    rng,
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
		doc := h.SingleLine
		if content.Kind == protocol.Markdown {
			doc = source.CommentToMarkdown(doc)
		}
		content.Value = doc
	case source.NoDocumentation:
		content.Value = signature
	case source.SynopsisDocumentation:
		if h.Synopsis != "" {
			doc := h.Synopsis
			if content.Kind == protocol.Markdown {
				doc = source.CommentToMarkdown(h.Synopsis)
			}
			content.Value = fmt.Sprintf("%s\n%s", doc, signature)
		} else {
			content.Value = signature
		}
	case source.FullDocumentation:
		if h.FullDocumentation != "" {
			doc := h.FullDocumentation
			if content.Kind == protocol.Markdown {
				doc = source.CommentToMarkdown(h.FullDocumentation)
			}
			content.Value = fmt.Sprintf("%s\n%s", signature, doc)
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
