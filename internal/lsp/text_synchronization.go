// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"fmt"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/lsp/telemetry/log"
	"golang.org/x/tools/internal/lsp/telemetry/trace"
	"golang.org/x/tools/internal/span"
)

func (s *Server) didOpen(ctx context.Context, params *protocol.DidOpenTextDocumentParams) error {
	uri := span.NewURI(params.TextDocument.URI)
	text := []byte(params.TextDocument.Text)

	// Confirm that the file's language ID is related to Go.
	fileKind := source.DetectLanguage(params.TextDocument.LanguageID, uri.Filename())

	// Open the file.
	s.session.DidOpen(ctx, uri, fileKind, text)

	// Run diagnostics on the newly-changed file.
	view := s.session.ViewOf(uri)
	go func() {
		ctx := view.BackgroundContext()
		ctx, done := trace.StartSpan(ctx, "lsp:background-worker")
		defer done()
		s.Diagnostics(ctx, view, uri)
	}()
	return nil
}

func (s *Server) didChange(ctx context.Context, params *protocol.DidChangeTextDocumentParams) error {
	if len(params.ContentChanges) < 1 {
		return jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "no content changes provided")
	}

	uri := span.NewURI(params.TextDocument.URI)

	// Check if the client sent the full content of the file.
	// We accept a full content change even if the server expected incremental changes.
	text, isFullChange := fullChange(params.ContentChanges)

	// We only accept an incremental change if the server expected it.
	if !isFullChange {
		switch s.textDocumentSyncKind {
		case protocol.Full:
			return fmt.Errorf("expected a full content change, received incremental changes for %s", uri)
		case protocol.Incremental:
			// Determine the new file content.
			var err error
			text, err = s.applyChanges(ctx, uri, params.ContentChanges)
			if err != nil {
				return err
			}
		}
	}
	// Cache the new file content and send fresh diagnostics.
	view := s.session.ViewOf(uri)
	if err := view.SetContent(ctx, uri, []byte(text)); err != nil {
		return err
	}
	// Run diagnostics on the newly-changed file.
	go func() {
		ctx := view.BackgroundContext()
		ctx, done := trace.StartSpan(ctx, "lsp:background-worker")
		defer done()
		s.Diagnostics(ctx, view, uri)
	}()
	return nil
}

func fullChange(changes []protocol.TextDocumentContentChangeEvent) (string, bool) {
	if len(changes) > 1 {
		return "", false
	}
	// The length of the changes must be 1 at this point.
	if changes[0].Range == nil && changes[0].RangeLength == 0 {
		return changes[0].Text, true
	}
	return "", false
}

func (s *Server) applyChanges(ctx context.Context, uri span.URI, changes []protocol.TextDocumentContentChangeEvent) (string, error) {
	content, _, err := s.session.GetFile(uri).Read(ctx)
	if err != nil {
		return "", jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "file not found")
	}
	fset := s.session.Cache().FileSet()
	for _, change := range changes {
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
	ctx = telemetry.File.With(ctx, uri)
	s.session.DidClose(uri)
	view := s.session.ViewOf(uri)
	if err := view.SetContent(ctx, uri, nil); err != nil {
		return err
	}
	clear := []span.URI{uri} // by default, clear the closed URI
	defer func() {
		for _, uri := range clear {
			if err := s.publishDiagnostics(ctx, view, uri, []source.Diagnostic{}); err != nil {
				log.Error(ctx, "failed to clear diagnostics", err, telemetry.File)
			}
		}
	}()
	// If the current file was the only open file for its package,
	// clear out all diagnostics for the package.
	f, err := view.GetFile(ctx, uri)
	if err != nil {
		log.Error(ctx, "no file for %s: %v", err, telemetry.File)
		return nil
	}
	// For non-Go files, don't return any diagnostics.
	gof, ok := f.(source.GoFile)
	if !ok {
		log.Error(ctx, "closing a non-Go file, no diagnostics to clear", nil, telemetry.File)
		return nil
	}
	pkg := gof.GetPackage(ctx)
	if pkg == nil {
		log.Error(ctx, "no package available", nil, telemetry.File)
		return nil
	}
	for _, filename := range pkg.GetFilenames() {
		// If other files from this package are open, don't clear.
		if s.session.IsOpen(span.NewURI(filename)) {
			clear = nil
			return nil
		}
		clear = append(clear, span.FileURI(filename))
	}
	return nil
}
