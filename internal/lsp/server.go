// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lsp implements LSP for gopls.
package lsp

import (
	"context"
	"sync"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/mod"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

const concurrentAnalyses = 1

// NewServer creates an LSP server and binds it to handle incoming client
// messages on on the supplied stream.
func NewServer(session source.Session, client protocol.Client) *Server {
	return &Server{
		delivered:       make(map[span.URI]sentDiagnostics),
		session:         session,
		client:          client,
		diagnosticsSema: make(chan struct{}, concurrentAnalyses),
	}
}

type serverState int

const (
	serverCreated      = serverState(iota)
	serverInitializing // set once the server has received "initialize" request
	serverInitialized  // set once the server has received "initialized" request
	serverShutDown
)

// Server implements the protocol.Server interface.
type Server struct {
	client protocol.Client

	stateMu sync.Mutex
	state   serverState

	session source.Session

	// changedFiles tracks files for which there has been a textDocument/didChange.
	changedFiles map[span.URI]struct{}

	// folders is only valid between initialize and initialized, and holds the
	// set of folders to build views for when we are ready
	pendingFolders []protocol.WorkspaceFolder

	// delivered is a cache of the diagnostics that the server has sent.
	deliveredMu sync.Mutex
	delivered   map[span.URI]sentDiagnostics

	showedInitialError   bool
	showedInitialErrorMu sync.Mutex

	// diagnosticsSema limits the concurrency of diagnostics runs, which can be expensive.
	diagnosticsSema chan struct{}
}

// sentDiagnostics is used to cache diagnostics that have been sent for a given file.
type sentDiagnostics struct {
	version      float64
	identifier   string
	sorted       []source.Diagnostic
	withAnalysis bool
	snapshotID   uint64
}

func (s *Server) cancelRequest(ctx context.Context, params *protocol.CancelParams) error {
	return nil
}

func (s *Server) codeLens(ctx context.Context, params *protocol.CodeLensParams) ([]protocol.CodeLens, error) {
	snapshot, fh, ok, err := s.beginFileRequest(params.TextDocument.URI, source.Mod)
	if !ok {
		return nil, err
	}
	return mod.CodeLens(ctx, snapshot, fh.Identity().URI)
}

func (s *Server) nonstandardRequest(ctx context.Context, method string, params interface{}) (interface{}, error) {
	paramMap := params.(map[string]interface{})
	if method == "gopls/diagnoseFiles" {
		for _, file := range paramMap["files"].([]interface{}) {
			snapshot, fh, ok, err := s.beginFileRequest(protocol.DocumentURI(file.(string)), source.UnknownKind)
			if !ok {
				return nil, err
			}

			fileID, diagnostics, err := source.FileDiagnostics(ctx, snapshot, fh.Identity().URI)
			if err != nil {
				return nil, err
			}
			if err := s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
				URI:         protocol.URIFromSpanURI(fh.Identity().URI),
				Diagnostics: toProtocolDiagnostics(diagnostics),
				Version:     fileID.Version,
			}); err != nil {
				return nil, err
			}
		}
		if err := s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
			URI: "gopls://diagnostics-done",
		}); err != nil {
			return nil, err
		}
		return struct{}{}, nil
	}
	return nil, notImplemented(method)
}

func notImplemented(method string) *jsonrpc2.Error {
	return jsonrpc2.NewErrorf(jsonrpc2.CodeMethodNotFound, "method %q not yet implemented", method)
}

//go:generate helper/helper -d protocol/tsserver.go -o server_gen.go -u .
