// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"sync"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

func (s *Server) initialize(ctx context.Context, params *protocol.ParamInitialize) (*protocol.InitializeResult, error) {
	s.stateMu.Lock()
	if s.state >= serverInitializing {
		defer s.stateMu.Unlock()
		return nil, errors.Errorf("%w: initialize called while server in %v state", jsonrpc2.ErrInvalidRequest, s.state)
	}
	s.state = serverInitializing
	s.stateMu.Unlock()

	s.clientPID = int(params.ProcessID)
	s.progress.supportsWorkDoneProgress = params.Capabilities.Window.WorkDoneProgress

	options := s.session.Options()
	defer func() { s.session.SetOptions(options) }()

	if err := s.handleOptionResults(ctx, source.SetOptions(options, params.InitializationOptions)); err != nil {
		return nil, err
	}
	options.ForClientCapabilities(params.Capabilities)

	folders := params.WorkspaceFolders
	if len(folders) == 0 {
		if params.RootURI != "" {
			folders = []protocol.WorkspaceFolder{{
				URI:  string(params.RootURI),
				Name: path.Base(params.RootURI.SpanURI().Filename()),
			}}
		}
	}
	for _, folder := range folders {
		uri := span.URIFromURI(folder.URI)
		if !uri.IsFile() {
			continue
		}
		s.pendingFolders = append(s.pendingFolders, folder)
	}
	// gopls only supports URIs with a file:// scheme, so if we have no
	// workspace folders with a supported scheme, fail to initialize.
	if len(folders) > 0 && len(s.pendingFolders) == 0 {
		return nil, fmt.Errorf("unsupported URI schemes: %v (gopls only supports file URIs)", folders)
	}

	var codeActionProvider interface{} = true
	if ca := params.Capabilities.TextDocument.CodeAction; len(ca.CodeActionLiteralSupport.CodeActionKind.ValueSet) > 0 {
		// If the client has specified CodeActionLiteralSupport,
		// send the code actions we support.
		//
		// Using CodeActionOptions is only valid if codeActionLiteralSupport is set.
		codeActionProvider = &protocol.CodeActionOptions{
			CodeActionKinds: s.getSupportedCodeActions(),
		}
	}
	var renameOpts interface{} = true
	if r := params.Capabilities.TextDocument.Rename; r.PrepareSupport {
		renameOpts = protocol.RenameOptions{
			PrepareProvider: r.PrepareSupport,
		}
	}

	goplsVersion, err := json.Marshal(debug.VersionInfo())
	if err != nil {
		return nil, err
	}

	return &protocol.InitializeResult{
		Capabilities: protocol.ServerCapabilities{
			InnerServerCapabilities: protocol.InnerServerCapabilities{
				CallHierarchyProvider: true,
				CodeActionProvider:    codeActionProvider,
				CompletionProvider: protocol.CompletionOptions{
					TriggerCharacters: []string{"."},
				},
				DefinitionProvider:         true,
				TypeDefinitionProvider:     true,
				ImplementationProvider:     true,
				DocumentFormattingProvider: true,
				DocumentSymbolProvider:     true,
				WorkspaceSymbolProvider:    true,
				ExecuteCommandProvider: protocol.ExecuteCommandOptions{
					Commands: options.SupportedCommands,
				},
				FoldingRangeProvider:      true,
				HoverProvider:             true,
				DocumentHighlightProvider: true,
				DocumentLinkProvider:      protocol.DocumentLinkOptions{},
				ReferencesProvider:        true,
				RenameProvider:            renameOpts,
				SignatureHelpProvider: protocol.SignatureHelpOptions{
					TriggerCharacters: []string{"(", ","},
				},
				TextDocumentSync: &protocol.TextDocumentSyncOptions{
					Change:    protocol.Incremental,
					OpenClose: true,
					Save: protocol.SaveOptions{
						IncludeText: false,
					},
				},
				Workspace: protocol.WorkspaceGn{
					WorkspaceFolders: protocol.WorkspaceFoldersGn{
						Supported:           true,
						ChangeNotifications: "workspace/didChangeWorkspaceFolders",
					},
				},
			},
		},
		ServerInfo: struct {
			Name    string `json:"name"`
			Version string `json:"version,omitempty"`
		}{
			Name:    "gopls",
			Version: string(goplsVersion),
		},
	}, nil
}

func (s *Server) initialized(ctx context.Context, params *protocol.InitializedParams) error {
	s.stateMu.Lock()
	if s.state >= serverInitialized {
		defer s.stateMu.Unlock()
		return errors.Errorf("%w: initialized called while server in %v state", jsonrpc2.ErrInvalidRequest, s.state)
	}
	s.state = serverInitialized
	s.stateMu.Unlock()

	for _, not := range s.notifications {
		s.client.ShowMessage(ctx, not)
	}
	s.notifications = nil

	options := s.session.Options()
	defer func() { s.session.SetOptions(options) }()

	if err := s.addFolders(ctx, s.pendingFolders); err != nil {
		return err
	}
	s.pendingFolders = nil

	if options.ConfigurationSupported && options.DynamicConfigurationSupported {
		registrations := []protocol.Registration{
			{
				ID:     "workspace/didChangeConfiguration",
				Method: "workspace/didChangeConfiguration",
			},
			{
				ID:     "workspace/didChangeWorkspaceFolders",
				Method: "workspace/didChangeWorkspaceFolders",
			},
		}
		if options.SemanticTokens {
			registrations = append(registrations, semanticTokenRegistration())
		}
		if err := s.client.RegisterCapability(ctx, &protocol.RegistrationParams{
			Registrations: registrations,
		}); err != nil {
			return err
		}
	}
	return nil
}

func (s *Server) addFolders(ctx context.Context, folders []protocol.WorkspaceFolder) error {
	originalViews := len(s.session.Views())
	viewErrors := make(map[span.URI]error)

	var wg sync.WaitGroup
	if s.session.Options().VerboseWorkDoneProgress {
		work := s.progress.start(ctx, DiagnosticWorkTitle(FromInitialWorkspaceLoad), "Calculating diagnostics for initial workspace load...", nil, nil)
		defer func() {
			go func() {
				wg.Wait()
				work.end("Done.")
			}()
		}()
	}
	// Only one view gets to have a workspace.
	assignedWorkspace := false
	var allFoldersWg sync.WaitGroup
	for _, folder := range folders {
		uri := span.URIFromURI(folder.URI)
		// Ignore non-file URIs.
		if !uri.IsFile() {
			continue
		}
		work := s.progress.start(ctx, "Setting up workspace", "Loading packages...", nil, nil)
		var workspaceURI span.URI = ""
		if !assignedWorkspace && s.clientPID != 0 {
			// For quick-and-dirty testing, set the temp workspace file to
			// $TMPDIR/gopls-<client PID>.workspace.
			//
			// This has a couple limitations:
			//  + If there are multiple workspace roots, only the first one gets
			//    written to this dir (and the client has no way to know precisely
			//    which one).
			//  + If a single client PID spawns multiple gopls sessions, they will
			//    clobber eachother's temp workspace.
			wsdir := filepath.Join(os.TempDir(), fmt.Sprintf("gopls-%d.workspace", s.clientPID))
			workspaceURI = span.URIFromPath(wsdir)
			assignedWorkspace = true
		}
		snapshot, release, err := s.addView(ctx, folder.Name, uri, workspaceURI)
		if err != nil {
			viewErrors[uri] = err
			work.end(fmt.Sprintf("Error loading packages: %s", err))
			continue
		}
		var swg sync.WaitGroup
		swg.Add(1)
		allFoldersWg.Add(1)
		go func() {
			defer swg.Done()
			defer allFoldersWg.Done()
			snapshot.AwaitInitialized(ctx)
			work.end("Finished loading packages.")
		}()

		// Print each view's environment.
		buf := &bytes.Buffer{}
		if err := snapshot.WriteEnv(ctx, buf); err != nil {
			viewErrors[uri] = err
			continue
		}
		event.Log(ctx, buf.String())

		// Diagnose the newly created view.
		wg.Add(1)
		go func() {
			s.diagnoseDetached(snapshot)
			swg.Wait()
			release()
			wg.Done()
		}()
	}

	// Register for file watching notifications, if they are supported.
	// Wait for all snapshots to be initialized first, since all files might
	// not yet be known to the snapshots.
	allFoldersWg.Wait()
	if err := s.updateWatchedDirectories(ctx); err != nil {
		event.Error(ctx, "failed to register for file watching notifications", err)
	}

	if len(viewErrors) > 0 {
		errMsg := fmt.Sprintf("Error loading workspace folders (expected %v, got %v)\n", len(folders), len(s.session.Views())-originalViews)
		for uri, err := range viewErrors {
			errMsg += fmt.Sprintf("failed to load view for %s: %v\n", uri, err)
		}
		return s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
			Type:    protocol.Error,
			Message: errMsg,
		})
	}
	return nil
}

// updateWatchedDirectories compares the current set of directories to watch
// with the previously registered set of directories. If the set of directories
// has changed, we unregister and re-register for file watching notifications.
// updatedSnapshots is the set of snapshots that have been updated.
func (s *Server) updateWatchedDirectories(ctx context.Context) error {
	patterns := s.session.FileWatchingGlobPatterns(ctx)

	s.watchedGlobPatternsMu.Lock()
	defer s.watchedGlobPatternsMu.Unlock()

	// Nothing to do if the set of workspace directories is unchanged.
	if equalURISet(s.watchedGlobPatterns, patterns) {
		return nil
	}

	// If the set of directories to watch has changed, register the updates and
	// unregister the previously watched directories. This ordering avoids a
	// period where no files are being watched. Still, if a user makes on-disk
	// changes before these updates are complete, we may miss them for the new
	// directories.
	prevID := s.watchRegistrationCount - 1
	if err := s.registerWatchedDirectoriesLocked(ctx, patterns); err != nil {
		return err
	}
	if prevID >= 0 {
		return s.client.UnregisterCapability(ctx, &protocol.UnregistrationParams{
			Unregisterations: []protocol.Unregistration{{
				ID:     watchedFilesCapabilityID(prevID),
				Method: "workspace/didChangeWatchedFiles",
			}},
		})
	}
	return nil
}

func watchedFilesCapabilityID(id int) string {
	return fmt.Sprintf("workspace/didChangeWatchedFiles-%d", id)
}

func equalURISet(m1, m2 map[string]struct{}) bool {
	if len(m1) != len(m2) {
		return false
	}
	for k := range m1 {
		_, ok := m2[k]
		if !ok {
			return false
		}
	}
	return true
}

// registerWatchedDirectoriesLocked sends the workspace/didChangeWatchedFiles
// registrations to the client and updates s.watchedDirectories.
func (s *Server) registerWatchedDirectoriesLocked(ctx context.Context, patterns map[string]struct{}) error {
	if !s.session.Options().DynamicWatchedFilesSupported {
		return nil
	}
	for k := range s.watchedGlobPatterns {
		delete(s.watchedGlobPatterns, k)
	}
	var watchers []protocol.FileSystemWatcher
	for pattern := range patterns {
		watchers = append(watchers, protocol.FileSystemWatcher{
			GlobPattern: pattern,
			Kind:        uint32(protocol.WatchChange + protocol.WatchDelete + protocol.WatchCreate),
		})
	}

	if err := s.client.RegisterCapability(ctx, &protocol.RegistrationParams{
		Registrations: []protocol.Registration{{
			ID:     watchedFilesCapabilityID(s.watchRegistrationCount),
			Method: "workspace/didChangeWatchedFiles",
			RegisterOptions: protocol.DidChangeWatchedFilesRegistrationOptions{
				Watchers: watchers,
			},
		}},
	}); err != nil {
		return err
	}
	s.watchRegistrationCount++

	for k, v := range patterns {
		s.watchedGlobPatterns[k] = v
	}
	return nil
}

func (s *Server) fetchConfig(ctx context.Context, name string, folder span.URI, o *source.Options) error {
	if !s.session.Options().ConfigurationSupported {
		return nil
	}
	configs, err := s.client.Configuration(ctx, &protocol.ParamConfiguration{
		ConfigurationParams: protocol.ConfigurationParams{
			Items: []protocol.ConfigurationItem{{
				ScopeURI: string(folder),
				Section:  "gopls",
			}},
		},
	})
	if err != nil {
		return fmt.Errorf("failed to get workspace configuration from client (%s): %v", folder, err)
	}
	for _, config := range configs {
		if err := s.handleOptionResults(ctx, source.SetOptions(o, config)); err != nil {
			return err
		}
	}
	return nil
}

func (s *Server) eventuallyShowMessage(ctx context.Context, msg *protocol.ShowMessageParams) error {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	if s.state == serverInitialized {
		return s.client.ShowMessage(ctx, msg)
	}
	s.notifications = append(s.notifications, msg)
	return nil
}

func (s *Server) handleOptionResults(ctx context.Context, results source.OptionResults) error {
	for _, result := range results {
		if result.Error != nil {
			msg := &protocol.ShowMessageParams{
				Type:    protocol.Error,
				Message: result.Error.Error(),
			}
			if err := s.eventuallyShowMessage(ctx, msg); err != nil {
				return err
			}
		}
		switch result.State {
		case source.OptionUnexpected:
			msg := &protocol.ShowMessageParams{
				Type:    protocol.Error,
				Message: fmt.Sprintf("unexpected gopls setting %q", result.Name),
			}
			if err := s.eventuallyShowMessage(ctx, msg); err != nil {
				return err
			}
		case source.OptionDeprecated:
			msg := fmt.Sprintf("gopls setting %q is deprecated", result.Name)
			if result.Replacement != "" {
				msg = fmt.Sprintf("%s, use %q instead", msg, result.Replacement)
			}
			if err := s.eventuallyShowMessage(ctx, &protocol.ShowMessageParams{
				Type:    protocol.Warning,
				Message: msg,
			}); err != nil {
				return err
			}
		}
	}
	return nil
}

// beginFileRequest checks preconditions for a file-oriented request and routes
// it to a snapshot.
// We don't want to return errors for benign conditions like wrong file type,
// so callers should do if !ok { return err } rather than if err != nil.
func (s *Server) beginFileRequest(ctx context.Context, pURI protocol.DocumentURI, expectKind source.FileKind) (source.Snapshot, source.VersionedFileHandle, bool, func(), error) {
	uri := pURI.SpanURI()
	if !uri.IsFile() {
		// Not a file URI. Stop processing the request, but don't return an error.
		return nil, nil, false, func() {}, nil
	}
	view, err := s.session.ViewOf(uri)
	if err != nil {
		return nil, nil, false, func() {}, err
	}
	snapshot, release := view.Snapshot(ctx)
	fh, err := snapshot.GetVersionedFile(ctx, uri)
	if err != nil {
		release()
		return nil, nil, false, func() {}, err
	}
	if expectKind != source.UnknownKind && fh.Kind() != expectKind {
		// Wrong kind of file. Nothing to do.
		release()
		return nil, nil, false, func() {}, nil
	}
	return snapshot, fh, true, release, nil
}

func (s *Server) shutdown(ctx context.Context) error {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	if s.state < serverInitialized {
		event.Log(ctx, "server shutdown without initialization")
	}
	if s.state != serverShutDown {
		// drop all the active views
		s.session.Shutdown(ctx)
		s.state = serverShutDown
	}
	return nil
}

func (s *Server) exit(ctx context.Context) error {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()

	// TODO: We need a better way to find the conn close method.
	s.client.(io.Closer).Close()

	if s.state != serverShutDown {
		// TODO: We should be able to do better than this.
		os.Exit(1)
	}
	// we don't terminate the process on a normal exit, we just allow it to
	// close naturally if needed after the connection is closed.
	return nil
}
