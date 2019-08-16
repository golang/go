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

type hoverKind int

const (
	singleLine = hoverKind(iota)
	noDocumentation
	synopsisDocumentation
	fullDocumentation

	// structured is an experimental setting that returns a structured hover format.
	// This format separates the signature from the documentation, so that the client
	// can do more manipulation of these fields.
	//
	// This should only be used by clients that support this behavior.
	structured
)

func (s *Server) hover(ctx context.Context, params *protocol.TextDocumentPositionParams) (*protocol.Hover, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.session.ViewOf(uri)
	f, err := getGoFile(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	m, err := getMapper(ctx, f)
	if err != nil {
		return nil, err
	}
	spn, err := m.PointSpan(params.Position)
	if err != nil {
		return nil, err
	}
	identRange, err := spn.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	ident, err := source.Identifier(ctx, f, identRange.Start)
	if err != nil {
		return nil, nil
	}
	hover, err := ident.Hover(ctx)
	if err != nil {
		return nil, err
	}
	identSpan, err := ident.Range.Span()
	if err != nil {
		return nil, err
	}
	rng, err := m.Range(identSpan)
	if err != nil {
		return nil, err
	}
	contents := s.toProtocolHoverContents(ctx, hover)
	return &protocol.Hover{
		Contents: contents,
		Range:    &rng,
	}, nil
}

func (s *Server) toProtocolHoverContents(ctx context.Context, h *source.HoverInformation) protocol.MarkupContent {
	content := protocol.MarkupContent{
		Kind: s.preferredContentFormat,
	}
	signature := h.Signature
	if content.Kind == protocol.Markdown {
		signature = fmt.Sprintf("```go\n%s\n```", h.Signature)
	}
	switch s.hoverKind {
	case singleLine:
		content.Value = h.SingleLine
	case noDocumentation:
		content.Value = signature
	case synopsisDocumentation:
		if h.Synopsis != "" {
			content.Value = fmt.Sprintf("%s\n%s", h.Synopsis, signature)
		} else {
			content.Value = signature
		}
	case fullDocumentation:
		if h.FullDocumentation != "" {
			content.Value = fmt.Sprintf("%s\n%s", signature, h.FullDocumentation)
		} else {
			content.Value = signature
		}
	case structured:
		b, err := json.Marshal(h)
		if err != nil {
			log.Error(ctx, "failed to marshal structured hover", err)
		}
		content.Value = string(b)
	}
	return content
}
