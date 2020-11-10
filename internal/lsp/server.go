// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lsp implements LSP for gopls.
package lsp

import (
	"context"
	"fmt"
	"sync"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

const concurrentAnalyses = 1

// NewServer creates an LSP server and binds it to handle incoming client
// messages on on the supplied stream.
func NewServer(session source.Session, client protocol.Client) *Server {
	return &Server{
		delivered:             make(map[span.URI]sentDiagnostics),
		gcOptimizationDetails: make(map[span.URI]struct{}),
		watchedDirectories:    make(map[span.URI]struct{}),
		changedFiles:          make(map[span.URI]struct{}),
		session:               session,
		client:                client,
		diagnosticsSema:       make(chan struct{}, concurrentAnalyses),
		progress:              newProgressTracker(client),
		debouncer:             newDebouncer(),
	}
}

type serverState int

const (
	serverCreated      = serverState(iota)
	serverInitializing // set once the server has received "initialize" request
	serverInitialized  // set once the server has received "initialized" request
	serverShutDown
)

func (s serverState) String() string {
	switch s {
	case serverCreated:
		return "created"
	case serverInitializing:
		return "initializing"
	case serverInitialized:
		return "initialized"
	case serverShutDown:
		return "shutDown"
	}
	return fmt.Sprintf("(unknown state: %d)", int(s))
}

// Server implements the protocol.Server interface.
type Server struct {
	client protocol.Client

	stateMu sync.Mutex
	state   serverState

	session   source.Session
	clientPID int

	// notifications generated before serverInitialized
	notifications []*protocol.ShowMessageParams

	// changedFiles tracks files for which there has been a textDocument/didChange.
	changedFilesMu sync.Mutex
	changedFiles   map[span.URI]struct{}

	// folders is only valid between initialize and initialized, and holds the
	// set of folders to build views for when we are ready
	pendingFolders []protocol.WorkspaceFolder

	// watchedDirectories is the set of directories that we have requested that
	// the client watch on disk. It will be updated as the set of directories
	// that the server should watch changes.
	watchedDirectoriesMu   sync.Mutex
	watchedDirectories     map[span.URI]struct{}
	watchRegistrationCount uint64

	// delivered is a cache of the diagnostics that the server has sent.
	deliveredMu sync.Mutex
	delivered   map[span.URI]sentDiagnostics

	// gcOptimizationDetails describes the packages for which we want
	// optimization details to be included in the diagnostics. The key is the
	// directory of the package.
	gcOptimizationDetailsMu sync.Mutex
	gcOptimizationDetails   map[span.URI]struct{}

	// diagnosticsSema limits the concurrency of diagnostics runs, which can be
	// expensive.
	diagnosticsSema chan struct{}

	progress *progressTracker

	// debouncer is used for debouncing diagnostics.
	debouncer *debouncer

	// When the workspace fails to load, we show its status through a progress
	// report with an error message.
	criticalErrorStatusMu sync.Mutex
	criticalErrorStatus   *workDone
}

// sentDiagnostics is used to cache diagnostics that have been sent for a given file.
type sentDiagnostics struct {
	id              source.VersionedFileIdentity
	sorted          []*source.Diagnostic
	includeAnalysis bool
	snapshotID      uint64
}

func (s *Server) workDoneProgressCancel(ctx context.Context, params *protocol.WorkDoneProgressCancelParams) error {
	return s.progress.cancel(ctx, params.Token)
}

func (s *Server) nonstandardRequest(ctx context.Context, method string, params interface{}) (interface{}, error) {
	paramMap := params.(map[string]interface{})
	if method == "gopls/diagnoseFiles" {
		for _, file := range paramMap["files"].([]interface{}) {
			snapshot, fh, ok, release, err := s.beginFileRequest(ctx, protocol.DocumentURI(file.(string)), source.UnknownKind)
			defer release()
			if !ok {
				return nil, err
			}

			fileID, diagnostics, err := source.FileDiagnostics(ctx, snapshot, fh.URI())
			if err != nil {
				return nil, err
			}
			if err := s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
				URI:         protocol.URIFromSpanURI(fh.URI()),
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

func notImplemented(method string) error {
	return errors.Errorf("%w: %q not yet implemented", jsonrpc2.ErrMethodNotFound, method)
}

//go:generate helper/helper -d protocol/tsserver.go -o server_gen.go -u .
