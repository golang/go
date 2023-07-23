// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run ./helper -d protocol/tsserver.go -o server_gen.go -u .

// Package lsp implements LSP for gopls.
package lsp

import (
	"context"
	"fmt"
	"os"
	"sync"

	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/progress"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/jsonrpc2"
)

const concurrentAnalyses = 1

// NewServer creates an LSP server and binds it to handle incoming client
// messages on the supplied stream.
func NewServer(session *cache.Session, client protocol.ClientCloser) *Server {
	return &Server{
		diagnostics:           map[span.URI]*fileReports{},
		gcOptimizationDetails: make(map[source.PackageID]struct{}),
		watchedGlobPatterns:   nil, // empty
		changedFiles:          make(map[span.URI]struct{}),
		session:               session,
		client:                client,
		diagnosticsSema:       make(chan struct{}, concurrentAnalyses),
		progress:              progress.NewTracker(client),
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
	client protocol.ClientCloser

	stateMu sync.Mutex
	state   serverState
	// notifications generated before serverInitialized
	notifications []*protocol.ShowMessageParams

	session *cache.Session

	tempDir string

	// changedFiles tracks files for which there has been a textDocument/didChange.
	changedFilesMu sync.Mutex
	changedFiles   map[span.URI]struct{}

	// folders is only valid between initialize and initialized, and holds the
	// set of folders to build views for when we are ready
	pendingFolders []protocol.WorkspaceFolder

	// watchedGlobPatterns is the set of glob patterns that we have requested
	// the client watch on disk. It will be updated as the set of directories
	// that the server should watch changes.
	// The map field may be reassigned but the map is immutable.
	watchedGlobPatternsMu  sync.Mutex
	watchedGlobPatterns    map[string]struct{}
	watchRegistrationCount int

	diagnosticsMu sync.Mutex
	diagnostics   map[span.URI]*fileReports

	// gcOptimizationDetails describes the packages for which we want
	// optimization details to be included in the diagnostics. The key is the
	// ID of the package.
	gcOptimizationDetailsMu sync.Mutex
	gcOptimizationDetails   map[source.PackageID]struct{}

	// diagnosticsSema limits the concurrency of diagnostics runs, which can be
	// expensive.
	diagnosticsSema chan struct{}

	progress *progress.Tracker

	// When the workspace fails to load, we show its status through a progress
	// report with an error message.
	criticalErrorStatusMu sync.Mutex
	criticalErrorStatus   *progress.WorkDone

	// Track an ongoing CPU profile created with the StartProfile command and
	// terminated with the StopProfile command.
	ongoingProfileMu sync.Mutex
	ongoingProfile   *os.File // if non-nil, an ongoing profile is writing to this file
}

func (s *Server) workDoneProgressCancel(ctx context.Context, params *protocol.WorkDoneProgressCancelParams) error {
	ctx, done := event.Start(ctx, "lsp.Server.workDoneProgressCancel")
	defer done()

	return s.progress.Cancel(params.Token)
}

func (s *Server) nonstandardRequest(ctx context.Context, method string, params interface{}) (interface{}, error) {
	ctx, done := event.Start(ctx, "lsp.Server.nonstandardRequest")
	defer done()

	switch method {
	case "gopls/diagnoseFiles":
		paramMap := params.(map[string]interface{})
		// TODO(adonovan): opt: parallelize FileDiagnostics(URI...), either
		// by calling it in multiple goroutines or, better, by making
		// the relevant APIs accept a set of URIs/packages.
		for _, file := range paramMap["files"].([]interface{}) {
			snapshot, fh, ok, release, err := s.beginFileRequest(ctx, protocol.DocumentURI(file.(string)), source.UnknownKind)
			defer release()
			if !ok {
				return nil, err
			}

			fileID, diagnostics, err := s.diagnoseFile(ctx, snapshot, fh.URI())
			if err != nil {
				return nil, err
			}
			if err := s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
				URI:         protocol.URIFromSpanURI(fh.URI()),
				Diagnostics: toProtocolDiagnostics(diagnostics),
				Version:     fileID.Version(),
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

// fileDiagnostics reports diagnostics in the specified file,
// as used by the "gopls check" or "gopls fix" commands.
//
// TODO(adonovan): opt: this function is called in a loop from the
// "gopls/diagnoseFiles" nonstandard request handler. It would be more
// efficient to compute the set of packages and TypeCheck and
// Analyze them all at once. Or instead support textDocument/diagnostic
// (golang/go#60122).
func (s *Server) diagnoseFile(ctx context.Context, snapshot source.Snapshot, uri span.URI) (source.FileHandle, []*source.Diagnostic, error) {
	fh, err := snapshot.ReadFile(ctx, uri)
	if err != nil {
		return nil, nil, err
	}
	pkg, _, err := source.NarrowestPackageForFile(ctx, snapshot, uri)
	if err != nil {
		return nil, nil, err
	}
	pkgDiags, err := pkg.DiagnosticsForFile(ctx, snapshot, uri)
	if err != nil {
		return nil, nil, err
	}
	adiags, err := source.Analyze(ctx, snapshot, map[source.PackageID]unit{pkg.Metadata().ID: {}}, nil /* progress tracker */)
	if err != nil {
		return nil, nil, err
	}
	var td, ad []*source.Diagnostic // combine load/parse/type + analysis diagnostics
	source.CombineDiagnostics(pkgDiags, adiags[uri], &td, &ad)
	s.storeDiagnostics(snapshot, uri, typeCheckSource, td, true)
	s.storeDiagnostics(snapshot, uri, analysisSource, ad, true)
	return fh, append(td, ad...), nil
}

func notImplemented(method string) error {
	return fmt.Errorf("%w: %q not yet implemented", jsonrpc2.ErrMethodNotFound, method)
}
