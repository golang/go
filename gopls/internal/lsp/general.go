// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"golang.org/x/tools/gopls/internal/lsp/debug"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/jsonrpc2"
)

func (s *Server) initialize(ctx context.Context, params *protocol.ParamInitialize) (*protocol.InitializeResult, error) {
	s.stateMu.Lock()
	if s.state >= serverInitializing {
		defer s.stateMu.Unlock()
		return nil, fmt.Errorf("%w: initialize called while server in %v state", jsonrpc2.ErrInvalidRequest, s.state)
	}
	s.state = serverInitializing
	s.stateMu.Unlock()

	// For uniqueness, use the gopls PID rather than params.ProcessID (the client
	// pid). Some clients might start multiple gopls servers, though they
	// probably shouldn't.
	pid := os.Getpid()
	s.tempDir = filepath.Join(os.TempDir(), fmt.Sprintf("gopls-%d.%s", pid, s.session.ID()))
	err := os.Mkdir(s.tempDir, 0700)
	if err != nil {
		// MkdirTemp could fail due to permissions issues. This is a problem with
		// the user's environment, but should not block gopls otherwise behaving.
		// All usage of s.tempDir should be predicated on having a non-empty
		// s.tempDir.
		event.Error(ctx, "creating temp dir", err)
		s.tempDir = ""
	}
	s.progress.SetSupportsWorkDoneProgress(params.Capabilities.Window.WorkDoneProgress)

	options := s.session.Options()
	defer func() { s.session.SetOptions(options) }()

	if err := s.handleOptionResults(ctx, source.SetOptions(options, params.InitializationOptions)); err != nil {
		return nil, err
	}
	options.ForClientCapabilities(params.Capabilities)

	if options.ShowBugReports {
		// Report the next bug that occurs on the server.
		bugCh := bug.Notify()
		go func() {
			b := <-bugCh
			msg := &protocol.ShowMessageParams{
				Type:    protocol.Error,
				Message: fmt.Sprintf("A bug occurred on the server: %s\nLocation:%s", b.Description, b.Key),
			}
			if err := s.eventuallyShowMessage(context.Background(), msg); err != nil {
				log.Printf("error showing bug: %v", err)
			}
		}()
	}

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

	versionInfo := debug.VersionInfo()

	// golang/go#45732: Warn users who've installed sergi/go-diff@v1.2.0, since
	// it will corrupt the formatting of their files.
	for _, dep := range versionInfo.Deps {
		if dep.Path == "github.com/sergi/go-diff" && dep.Version == "v1.2.0" {
			if err := s.eventuallyShowMessage(ctx, &protocol.ShowMessageParams{
				Message: `It looks like you have a bad gopls installation.
Please reinstall gopls by running 'GO111MODULE=on go install golang.org/x/tools/gopls@latest'.
See https://github.com/golang/go/issues/45732 for more information.`,
				Type: protocol.Error,
			}); err != nil {
				return nil, err
			}
		}
	}

	goplsVersion, err := json.Marshal(versionInfo)
	if err != nil {
		return nil, err
	}

	return &protocol.InitializeResult{
		Capabilities: protocol.ServerCapabilities{
			CallHierarchyProvider: true,
			CodeActionProvider:    codeActionProvider,
			CodeLensProvider:      &protocol.CodeLensOptions{}, // must be non-nil to enable the code lens capability
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
			InlayHintProvider:         protocol.InlayHintOptions{},
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
			Workspace: protocol.Workspace6Gn{
				WorkspaceFolders: protocol.WorkspaceFolders5Gn{
					Supported:           true,
					ChangeNotifications: "workspace/didChangeWorkspaceFolders",
				},
			},
		},
		ServerInfo: protocol.PServerInfoMsg_initialize{
			Name:    "gopls",
			Version: string(goplsVersion),
		},
	}, nil
}

func (s *Server) initialized(ctx context.Context, params *protocol.InitializedParams) error {
	s.stateMu.Lock()
	if s.state >= serverInitialized {
		defer s.stateMu.Unlock()
		return fmt.Errorf("%w: initialized called while server in %v state", jsonrpc2.ErrInvalidRequest, s.state)
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
	s.checkViewGoVersions()

	var registrations []protocol.Registration
	if options.ConfigurationSupported && options.DynamicConfigurationSupported {
		registrations = append(registrations, protocol.Registration{
			ID:     "workspace/didChangeConfiguration",
			Method: "workspace/didChangeConfiguration",
		})
	}
	if options.SemanticTokens && options.DynamicRegistrationSemanticTokensSupported {
		registrations = append(registrations, semanticTokenRegistration(options.SemanticTypes, options.SemanticMods))
	}
	if len(registrations) > 0 {
		if err := s.client.RegisterCapability(ctx, &protocol.RegistrationParams{
			Registrations: registrations,
		}); err != nil {
			return err
		}
	}
	return nil
}

// GoVersionTable maps Go versions to the gopls version in which support will
// be deprecated, and the final gopls version supporting them without warnings.
// Keep this in sync with gopls/README.md
//
// Must be sorted in ascending order of Go version.
//
// Mutable for testing.
var GoVersionTable = []GoVersionSupport{
	{12, "", "v0.7.5"},
	{15, "v0.11.0", "v0.9.5"},
}

// GoVersionSupport holds information about end-of-life Go version support.
type GoVersionSupport struct {
	GoVersion           int
	DeprecatedVersion   string // if unset, the version is already deprecated
	InstallGoplsVersion string
}

// OldestSupportedGoVersion is the last X in Go 1.X that this version of gopls
// supports.
func OldestSupportedGoVersion() int {
	return GoVersionTable[len(GoVersionTable)-1].GoVersion + 1
}

// versionMessage returns the warning/error message to display if the user is
// on the given Go version, if any. The goVersion variable is the X in Go 1.X.
//
// If goVersion is invalid (< 0), it returns "", 0.
func versionMessage(goVersion int) (string, protocol.MessageType) {
	if goVersion < 0 {
		return "", 0
	}

	for _, v := range GoVersionTable {
		if goVersion <= v.GoVersion {
			var msgBuilder strings.Builder

			mType := protocol.Error
			fmt.Fprintf(&msgBuilder, "Found Go version 1.%d", goVersion)
			if v.DeprecatedVersion != "" {
				// not deprecated yet, just a warning
				fmt.Fprintf(&msgBuilder, ", which will be unsupported by gopls %s. ", v.DeprecatedVersion)
				mType = protocol.Warning
			} else {
				fmt.Fprint(&msgBuilder, ", which is not supported by this version of gopls. ")
			}
			fmt.Fprintf(&msgBuilder, "Please upgrade to Go 1.%d or later and reinstall gopls. ", OldestSupportedGoVersion())
			fmt.Fprintf(&msgBuilder, "If you can't upgrade and want this message to go away, please install gopls %s. ", v.InstallGoplsVersion)
			fmt.Fprint(&msgBuilder, "See https://go.dev/s/gopls-support-policy for more details.")

			return msgBuilder.String(), mType
		}
	}
	return "", 0
}

// checkViewGoVersions checks whether any Go version used by a view is too old,
// raising a showMessage notification if so.
//
// It should be called after views change.
func (s *Server) checkViewGoVersions() {
	oldestVersion := -1
	for _, view := range s.session.Views() {
		viewVersion := view.GoVersion()
		if oldestVersion == -1 || viewVersion < oldestVersion {
			oldestVersion = viewVersion
		}
	}

	if msg, mType := versionMessage(oldestVersion); msg != "" {
		s.eventuallyShowMessage(context.Background(), &protocol.ShowMessageParams{
			Type:    mType,
			Message: msg,
		})
	}
}

func (s *Server) addFolders(ctx context.Context, folders []protocol.WorkspaceFolder) error {
	originalViews := len(s.session.Views())
	viewErrors := make(map[span.URI]error)

	var ndiagnose sync.WaitGroup // number of unfinished diagnose calls
	if s.session.Options().VerboseWorkDoneProgress {
		work := s.progress.Start(ctx, DiagnosticWorkTitle(FromInitialWorkspaceLoad), "Calculating diagnostics for initial workspace load...", nil, nil)
		defer func() {
			go func() {
				ndiagnose.Wait()
				work.End(ctx, "Done.")
			}()
		}()
	}
	// Only one view gets to have a workspace.
	var nsnapshots sync.WaitGroup // number of unfinished snapshot initializations
	for _, folder := range folders {
		uri := span.URIFromURI(folder.URI)
		// Ignore non-file URIs.
		if !uri.IsFile() {
			continue
		}
		work := s.progress.Start(ctx, "Setting up workspace", "Loading packages...", nil, nil)
		snapshot, release, err := s.addView(ctx, folder.Name, uri)
		if err != nil {
			if err == source.ErrViewExists {
				continue
			}
			viewErrors[uri] = err
			work.End(ctx, fmt.Sprintf("Error loading packages: %s", err))
			continue
		}
		// Inv: release() must be called once.

		// Print each view's environment.
		var buf bytes.Buffer
		if err := snapshot.WriteEnv(ctx, &buf); err != nil {
			viewErrors[uri] = err
			release()
			continue
		}
		event.Log(ctx, buf.String())

		// Initialize snapshot asynchronously.
		initialized := make(chan struct{})
		nsnapshots.Add(1)
		go func() {
			snapshot.AwaitInitialized(ctx)
			work.End(ctx, "Finished loading packages.")
			nsnapshots.Done()
			close(initialized) // signal
		}()

		// Diagnose the newly created view asynchronously.
		ndiagnose.Add(1)
		go func() {
			s.diagnoseDetached(snapshot)
			<-initialized
			release()
			ndiagnose.Done()
		}()
	}

	// Wait for snapshots to be initialized so that all files are known.
	// (We don't need to wait for diagnosis to finish.)
	nsnapshots.Wait()

	// Register for file watching notifications, if they are supported.
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
		Items: []protocol.ConfigurationItem{{
			ScopeURI: string(folder),
			Section:  "gopls",
		}},
	},
	)
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
	var warnings, errors []string
	for _, result := range results {
		switch result.Error.(type) {
		case nil:
			// nothing to do
		case *source.SoftError:
			warnings = append(warnings, result.Error.Error())
		default:
			errors = append(errors, result.Error.Error())
		}
	}

	// Sort messages, but put errors first.
	//
	// Having stable content for the message allows clients to de-duplicate. This
	// matters because we may send duplicate warnings for clients that support
	// dynamic configuration: one for the initial settings, and then more for the
	// individual view settings.
	var msgs []string
	msgType := protocol.Warning
	if len(errors) > 0 {
		msgType = protocol.Error
		sort.Strings(errors)
		msgs = append(msgs, errors...)
	}
	if len(warnings) > 0 {
		sort.Strings(warnings)
		msgs = append(msgs, warnings...)
	}

	if len(msgs) > 0 {
		// Settings
		combined := "Invalid settings: " + strings.Join(msgs, "; ")
		params := &protocol.ShowMessageParams{
			Type:    msgType,
			Message: combined,
		}
		return s.eventuallyShowMessage(ctx, params)
	}

	return nil
}

// beginFileRequest checks preconditions for a file-oriented request and routes
// it to a snapshot.
// We don't want to return errors for benign conditions like wrong file type,
// so callers should do if !ok { return err } rather than if err != nil.
// The returned cleanup function is non-nil even in case of false/error result.
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
	if expectKind != source.UnknownKind && view.FileKind(fh) != expectKind {
		// Wrong kind of file. Nothing to do.
		release()
		return nil, nil, false, func() {}, nil
	}
	return snapshot, fh, true, release, nil
}

// shutdown implements the 'shutdown' LSP handler. It releases resources
// associated with the server and waits for all ongoing work to complete.
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
		if s.tempDir != "" {
			if err := os.RemoveAll(s.tempDir); err != nil {
				event.Error(ctx, "removing temp dir", err)
			}
		}
	}
	return nil
}

func (s *Server) exit(ctx context.Context) error {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()

	s.client.Close()

	if s.state != serverShutDown {
		// TODO: We should be able to do better than this.
		os.Exit(1)
	}
	// we don't terminate the process on a normal exit, we just allow it to
	// close naturally if needed after the connection is closed.
	return nil
}
