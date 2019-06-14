// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func (s *Server) didOpen(ctx context.Context, params *protocol.DidOpenTextDocumentParams) error {
	uri := span.NewURI(params.TextDocument.URI)
	s.session.DidOpen(uri)
	return s.cacheAndDiagnose(ctx, uri, []byte(params.TextDocument.Text))
}

func (s *Server) didChange(ctx context.Context, params *protocol.DidChangeTextDocumentParams) error {
	if len(params.ContentChanges) < 1 {
		return jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "no content changes provided")
	}

	var text string
	switch s.textDocumentSyncKind {
	case protocol.Incremental:
		var err error
		text, err = s.applyChanges(ctx, params)
		if err != nil {
			return err
		}
	case protocol.Full:
		// We expect the full content of file, i.e. a single change with no range.
		change := params.ContentChanges[0]
		if change.RangeLength != 0 {
			return jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "unexpected change range provided")
		}
		text = change.Text
	}
	return s.cacheAndDiagnose(ctx, span.NewURI(params.TextDocument.URI), []byte(text))
}

func (s *Server) cacheAndDiagnose(ctx context.Context, uri span.URI, content []byte) error {
	view := s.session.ViewOf(uri)
	if err := view.SetContent(ctx, uri, content); err != nil {
		return err
	}
	go func() {
		ctx := view.BackgroundContext()
		s.Diagnostics(ctx, view, uri)
	}()
	return nil
}

func (s *Server) applyChanges(ctx context.Context, params *protocol.DidChangeTextDocumentParams) (string, error) {
	if len(params.ContentChanges) == 1 && params.ContentChanges[0].Range == nil {
		// If range is empty, we expect the full content of file, i.e. a single change with no range.
		change := params.ContentChanges[0]
		if change.RangeLength != 0 {
			return "", jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "unexpected change range provided")
		}
		return change.Text, nil
	}

	uri := span.NewURI(params.TextDocument.URI)
	content, _, err := s.session.GetFile(uri).Read(ctx)
	if err != nil {
		return "", jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "file not found")
	}
	fset := s.session.Cache().FileSet()
	for _, change := range params.ContentChanges {
		// Update column mapper along with the content.
		m := protocol.NewColumnMapper(uri, uri.Filename(), fset, nil, content)

		spn, err := m.RangeSpan(*change.Range)
		if err != nil {
			return "", err
		}
		if !spn.HasOffset() {
			return "", jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "invalid range for content change")
		}
		start, end := spn.Start().Offset(), spn.End().Offset()
		if end < start {
			return "", jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "invalid range for content change")
		}
		var buf bytes.Buffer
		buf.Write(content[:start])
		buf.WriteString(change.Text)
		buf.Write(content[end:])
		content = buf.Bytes()
	}
	return string(content), nil
}

func (s *Server) didSave(ctx context.Context, params *protocol.DidSaveTextDocumentParams) error {
	s.session.DidSave(span.NewURI(params.TextDocument.URI))
	return nil
}

func (s *Server) didClose(ctx context.Context, params *protocol.DidCloseTextDocumentParams) error {
	uri := span.NewURI(params.TextDocument.URI)
	s.session.DidClose(uri)
	view := s.session.ViewOf(uri)
	if err := view.SetContent(ctx, uri, nil); err != nil {
		return err
	}
	// If the current file was the only open file for its package,
	// clear out all diagnostics for the package.
	f, err := view.GetFile(ctx, uri)
	if err != nil {
		s.session.Logger().Errorf(ctx, "no file for %s: %v", uri, err)
		return nil
	}
	// For non-Go files, don't return any diagnostics.
	gof, ok := f.(source.GoFile)
	if !ok {
		return nil
	}
	pkg := gof.GetPackage(ctx)
	if pkg == nil {
		s.session.Logger().Errorf(ctx, "no package available for %s", uri)
		return nil
	}
	reports := make(map[span.URI][]source.Diagnostic)
	clearDiagnostics := true
	for _, filename := range pkg.GetFilenames() {
		uri := span.NewURI(filename)
		reports[uri] = []source.Diagnostic{}
		if span.CompareURI(uri, gof.URI()) == 0 {
			continue
		}
		// If other files from this package are open.
		if s.session.IsOpen(uri) {
			clearDiagnostics = false
		}
	}
	// We still have open files for this package, so don't clear diagnostics.
	if !clearDiagnostics {
		return nil
	}
	for uri, diagnostics := range reports {
		if err := s.publishDiagnostics(ctx, view, uri, diagnostics); err != nil {
			s.session.Logger().Errorf(ctx, "failed to clear diagnostics for %s: %v", uri, err)
		}
	}
	return nil
}
