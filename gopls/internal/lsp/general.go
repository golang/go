// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"encoding/json"
	"fmt"
	"go/build"
	"log"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/debug"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/gopls/internal/telemetry"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/jsonrpc2"
)

func (s *Server) initialize(ctx context.Context, params *protocol.ParamInitialize) (*protocol.InitializeResult, error) {
	ctx, done := event.Start(ctx, "lsp.Server.initialize")
	defer done()

	telemetry.RecordClientInfo(params)

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

	options := s.Options().Clone()
	// TODO(rfindley): remove the error return from handleOptionResults, and
	// eliminate this defer.
	defer func() { s.SetOptions(options) }()

	if err := s.handleOptionResults(ctx, source.SetOptions(options, params.InitializationOptions)); err != nil {
		return nil, err
	}
	options.ForClientCapabilities(params.ClientInfo, params.Capabilities)

	if options.ShowBugReports {
		// Report the next bug that occurs on the server.
		bug.Handle(func(b bug.Bug) {
			msg := &protocol.ShowMessageParams{
				Type:    protocol.Error,
				Message: fmt.Sprintf("A bug occurred on the server: %s\nLocation:%s", b.Description, b.Key),
			}
			go func() {
				if err := s.eventuallyShowMessage(context.Background(), msg); err != nil {
					log.Printf("error showing bug: %v", err)
				}
			}()
		})
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
	if r := params.Capabilities.TextDocument.Rename; r != nil && r.PrepareSupport {
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
			CallHierarchyProvider: &protocol.Or_ServerCapabilities_callHierarchyProvider{Value: true},
			CodeActionProvider:    codeActionProvider,
			CodeLensProvider:      &protocol.CodeLensOptions{}, // must be non-nil to enable the code lens capability
			CompletionProvider: &protocol.CompletionOptions{
				TriggerCharacters: []string{"."},
			},
			DefinitionProvider:         &protocol.Or_ServerCapabilities_definitionProvider{Value: true},
			TypeDefinitionProvider:     &protocol.Or_ServerCapabilities_typeDefinitionProvider{Value: true},
			ImplementationProvider:     &protocol.Or_ServerCapabilities_implementationProvider{Value: true},
			DocumentFormattingProvider: &protocol.Or_ServerCapabilities_documentFormattingProvider{Value: true},
			DocumentSymbolProvider:     &protocol.Or_ServerCapabilities_documentSymbolProvider{Value: true},
			WorkspaceSymbolProvider:    &protocol.Or_ServerCapabilities_workspaceSymbolProvider{Value: true},
			ExecuteCommandProvider: &protocol.ExecuteCommandOptions{
				Commands: nonNilSliceString(options.SupportedCommands),
			},
			FoldingRangeProvider:      &protocol.Or_ServerCapabilities_foldingRangeProvider{Value: true},
			HoverProvider:             &protocol.Or_ServerCapabilities_hoverProvider{Value: true},
			DocumentHighlightProvider: &protocol.Or_ServerCapabilities_documentHighlightProvider{Value: true},
			DocumentLinkProvider:      &protocol.DocumentLinkOptions{},
			InlayHintProvider:         protocol.InlayHintOptions{},
			ReferencesProvider:        &protocol.Or_ServerCapabilities_referencesProvider{Value: true},
			RenameProvider:            renameOpts,
			SelectionRangeProvider:    &protocol.Or_ServerCapabilities_selectionRangeProvider{Value: true},
			SemanticTokensProvider: protocol.SemanticTokensOptions{
				Range: &protocol.Or_SemanticTokensOptions_range{Value: true},
				Full:  &protocol.Or_SemanticTokensOptions_full{Value: true},
				Legend: protocol.SemanticTokensLegend{
					TokenTypes:     nonNilSliceString(options.SemanticTypes),
					TokenModifiers: nonNilSliceString(options.SemanticMods),
				},
			},
			SignatureHelpProvider: &protocol.SignatureHelpOptions{
				TriggerCharacters: []string{"(", ","},
			},
			TextDocumentSync: &protocol.TextDocumentSyncOptions{
				Change:    protocol.Incremental,
				OpenClose: true,
				Save: &protocol.SaveOptions{
					IncludeText: false,
				},
			},
			Workspace: &protocol.Workspace6Gn{
				WorkspaceFolders: &protocol.WorkspaceFolders5Gn{
					Supported:           true,
					ChangeNotifications: "workspace/didChangeWorkspaceFolders",
				},
			},
		},
		ServerInfo: &protocol.PServerInfoMsg_initialize{
			Name:    "gopls",
			Version: string(goplsVersion),
		},
	}, nil
}

func (s *Server) initialized(ctx context.Context, params *protocol.InitializedParams) error {
	ctx, done := event.Start(ctx, "lsp.Server.initialized")
	defer done()

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

	if err := s.addFolders(ctx, s.pendingFolders); err != nil {
		return err
	}
	s.pendingFolders = nil
	s.checkViewGoVersions()

	var registrations []protocol.Registration
	options := s.Options()
	if options.ConfigurationSupported && options.DynamicConfigurationSupported {
		registrations = append(registrations, protocol.Registration{
			ID:     "workspace/didChangeConfiguration",
			Method: "workspace/didChangeConfiguration",
		})
	}
	if len(registrations) > 0 {
		if err := s.client.RegisterCapability(ctx, &protocol.RegistrationParams{
			Registrations: registrations,
		}); err != nil {
			return err
		}
	}

	// Ask (maybe) about enabling telemetry. Do this asynchronously, as it's OK
	// for users to ignore or dismiss the question.
	go s.maybePromptForTelemetry(ctx, options.TelemetryPrompt)

	return nil
}

// GoVersionTable maps Go versions to the gopls version in which support will
// be deprecated, and the final gopls version supporting them without warnings.
// Keep this in sync with gopls/README.md.
//
// Must be sorted in ascending order of Go version.
//
// Mutable for testing.
var GoVersionTable = []GoVersionSupport{
	{12, "", "v0.7.5"},
	{15, "", "v0.9.5"},
	{16, "v0.13.0", "v0.11.0"},
	{17, "v0.13.0", "v0.11.0"},
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

// versionMessage returns the warning/error message to display if the user has
// the given Go version, if any. The goVersion variable is the X in Go 1.X. If
// fromBuild is set, the Go version is the version used to build gopls.
// Otherwise, it is the go command version.
//
// If goVersion is invalid (< 0), it returns "", 0.
func versionMessage(goVersion int, fromBuild bool) (string, protocol.MessageType) {
	if goVersion < 0 {
		return "", 0
	}

	for _, v := range GoVersionTable {
		if goVersion <= v.GoVersion {
			var msgBuilder strings.Builder

			mType := protocol.Error
			if fromBuild {
				fmt.Fprintf(&msgBuilder, "Gopls was built with Go version 1.%d", goVersion)
			} else {
				fmt.Fprintf(&msgBuilder, "Found Go version 1.%d", goVersion)
			}
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
	oldestVersion, fromBuild := go1Point(), true
	for _, view := range s.session.Views() {
		viewVersion := view.GoVersion()
		if oldestVersion == -1 || viewVersion < oldestVersion {
			oldestVersion, fromBuild = viewVersion, false
		}
		telemetry.RecordViewGoVersion(viewVersion)
	}

	if msg, mType := versionMessage(oldestVersion, fromBuild); msg != "" {
		s.eventuallyShowMessage(context.Background(), &protocol.ShowMessageParams{
			Type:    mType,
			Message: msg,
		})
	}
}

// go1Point returns the x in Go 1.x. If an error occurs extracting the go
// version, it returns -1.
//
// Copied from the testenv package.
func go1Point() int {
	for i := len(build.Default.ReleaseTags) - 1; i >= 0; i-- {
		var version int
		if _, err := fmt.Sscanf(build.Default.ReleaseTags[i], "go1.%d", &version); err != nil {
			continue
		}
		return version
	}
	return -1
}

func (s *Server) addFolders(ctx context.Context, folders []protocol.WorkspaceFolder) error {
	originalViews := len(s.session.Views())
	viewErrors := make(map[span.URI]error)

	var ndiagnose sync.WaitGroup // number of unfinished diagnose calls
	if s.Options().VerboseWorkDoneProgress {
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
			s.diagnoseSnapshot(snapshot, nil, false, 0)
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
// The caller must not subsequently mutate patterns.
func (s *Server) registerWatchedDirectoriesLocked(ctx context.Context, patterns map[string]struct{}) error {
	if !s.Options().DynamicWatchedFilesSupported {
		return nil
	}
	s.watchedGlobPatterns = patterns
	watchers := make([]protocol.FileSystemWatcher, 0, len(patterns)) // must be a slice
	val := protocol.WatchChange | protocol.WatchDelete | protocol.WatchCreate
	for pattern := range patterns {
		watchers = append(watchers, protocol.FileSystemWatcher{
			GlobPattern: pattern,
			Kind:        &val,
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
	return nil
}

// Options returns the current server options.
//
// The caller must not modify the result.
func (s *Server) Options() *source.Options {
	s.optionsMu.Lock()
	defer s.optionsMu.Unlock()
	return s.options
}

// SetOptions sets the current server options.
//
// The caller must not subsequently modify the options.
func (s *Server) SetOptions(opts *source.Options) {
	s.optionsMu.Lock()
	defer s.optionsMu.Unlock()
	s.options = opts
}

func (s *Server) fetchFolderOptions(ctx context.Context, folder span.URI) (*source.Options, error) {
	if opts := s.Options(); !opts.ConfigurationSupported {
		return opts, nil
	}
	configs, err := s.client.Configuration(ctx, &protocol.ParamConfiguration{
		Items: []protocol.ConfigurationItem{{
			ScopeURI: string(folder),
			Section:  "gopls",
		}},
	},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to get workspace configuration from client (%s): %v", folder, err)
	}

	folderOpts := s.Options().Clone()
	for _, config := range configs {
		if err := s.handleOptionResults(ctx, source.SetOptions(folderOpts, config)); err != nil {
			return nil, err
		}
	}
	return folderOpts, nil
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
func (s *Server) beginFileRequest(ctx context.Context, pURI protocol.DocumentURI, expectKind source.FileKind) (source.Snapshot, source.FileHandle, bool, func(), error) {
	uri := pURI.SpanURI()
	if !uri.IsFile() {
		// Not a file URI. Stop processing the request, but don't return an error.
		return nil, nil, false, func() {}, nil
	}
	view, err := s.session.ViewOf(uri)
	if err != nil {
		return nil, nil, false, func() {}, err
	}
	snapshot, release, err := view.Snapshot()
	if err != nil {
		return nil, nil, false, func() {}, err
	}
	fh, err := snapshot.ReadFile(ctx, uri)
	if err != nil {
		release()
		return nil, nil, false, func() {}, err
	}
	if expectKind != source.UnknownKind && snapshot.FileKind(fh) != expectKind {
		// Wrong kind of file. Nothing to do.
		release()
		return nil, nil, false, func() {}, nil
	}
	return snapshot, fh, true, release, nil
}

// shutdown implements the 'shutdown' LSP handler. It releases resources
// associated with the server and waits for all ongoing work to complete.
func (s *Server) shutdown(ctx context.Context) error {
	ctx, done := event.Start(ctx, "lsp.Server.shutdown")
	defer done()

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
	ctx, done := event.Start(ctx, "lsp.Server.exit")
	defer done()

	s.stateMu.Lock()
	defer s.stateMu.Unlock()

	s.client.Close()

	if s.state != serverShutDown {
		// TODO: We should be able to do better than this.
		os.Exit(1)
	}
	// We don't terminate the process on a normal exit, we just allow it to
	// close naturally if needed after the connection is closed.
	return nil
}

// TODO: when we can assume go1.18, replace with generic
// (after retiring support for go1.17)
func nonNilSliceString(x []string) []string {
	if x == nil {
		return []string{}
	}
	return x
}
func nonNilSliceTextEdit(x []protocol.TextEdit) []protocol.TextEdit {
	if x == nil {
		return []protocol.TextEdit{}
	}

	return x
}
func nonNilSliceCompletionItemTag(x []protocol.CompletionItemTag) []protocol.CompletionItemTag {
	if x == nil {
		return []protocol.CompletionItemTag{}
	}
	return x
}
func emptySliceDiagnosticTag(x []protocol.DiagnosticTag) []protocol.DiagnosticTag {
	if x == nil {
		return []protocol.DiagnosticTag{}
	}
	return x
}
