// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"path/filepath"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/xcontext"
	errors "golang.org/x/xerrors"
)

func (s *Server) executeCommand(ctx context.Context, params *protocol.ExecuteCommandParams) (interface{}, error) {
	var command *source.Command
	for _, c := range source.Commands {
		if c.ID() == params.Command {
			command = c
			break
		}
	}
	if command == nil {
		return nil, fmt.Errorf("no known command")
	}
	var match bool
	for _, name := range s.session.Options().SupportedCommands {
		if command.ID() == name {
			match = true
			break
		}
	}
	if !match {
		return nil, fmt.Errorf("%s is not a supported command", command.ID())
	}
	// Some commands require that all files are saved to disk. If we detect
	// unsaved files, warn the user instead of running the commands.
	unsaved := false
	for _, overlay := range s.session.Overlays() {
		if !overlay.Saved() {
			unsaved = true
			break
		}
	}
	if unsaved {
		switch params.Command {
		case source.CommandTest.ID(),
			source.CommandGenerate.ID(),
			source.CommandToggleDetails.ID(),
			source.CommandAddDependency.ID(),
			source.CommandUpgradeDependency.ID(),
			source.CommandRemoveDependency.ID(),
			source.CommandVendor.ID():
			// TODO(PJW): for Toggle, not an error if it is being disabled
			err := errors.New("All files must be saved first")
			s.showCommandError(ctx, command.Title, err)
			return nil, nil
		}
	}
	ctx, cancel := context.WithCancel(xcontext.Detach(ctx))

	var work *workDone
	// Don't show progress for suggested fixes. They should be quick.
	if !command.IsSuggestedFix() {
		// Start progress prior to spinning off a goroutine specifically so that
		// clients are aware of the work item before the command completes. This
		// matters for regtests, where having a continuous thread of work is
		// convenient for assertions.
		work = s.progress.start(ctx, command.Title, "Running...", params.WorkDoneToken, cancel)
	}

	run := func() {
		defer cancel()
		err := s.runCommand(ctx, work, command, params.Arguments)
		switch {
		case errors.Is(err, context.Canceled):
			work.end(command.Title + ": canceled")
		case err != nil:
			event.Error(ctx, fmt.Sprintf("%s: command error", command.Title), err)
			work.end(command.Title + ": failed")
			// Show a message when work completes with error, because the progress end
			// message is typically dismissed immediately by LSP clients.
			s.showCommandError(ctx, command.Title, err)
		default:
			work.end(command.ID() + ": completed")
		}
	}
	if command.Async {
		go run()
	} else {
		run()
	}
	// Errors running the command are displayed to the user above, so don't
	// return them.
	return nil, nil
}

func (s *Server) runSuggestedFixCommand(ctx context.Context, command *source.Command, args []json.RawMessage) error {
	var uri protocol.DocumentURI
	var rng protocol.Range
	if err := source.UnmarshalArgs(args, &uri, &rng); err != nil {
		return err
	}
	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, uri, source.Go)
	defer release()
	if !ok {
		return err
	}
	edits, err := command.SuggestedFix(ctx, snapshot, fh, rng)
	if err != nil {
		return err
	}
	r, err := s.client.ApplyEdit(ctx, &protocol.ApplyWorkspaceEditParams{
		Edit: protocol.WorkspaceEdit{
			DocumentChanges: edits,
		},
	})
	if err != nil {
		return err
	}
	if !r.Applied {
		return errors.New(r.FailureReason)
	}
	return nil
}

func (s *Server) showCommandError(ctx context.Context, title string, err error) {
	// Command error messages should not be cancelable.
	ctx = xcontext.Detach(ctx)
	if err := s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
		Type:    protocol.Error,
		Message: fmt.Sprintf("%s failed: %v", title, err),
	}); err != nil {
		event.Error(ctx, title+": failed to show message", err)
	}
}

func (s *Server) runCommand(ctx context.Context, work *workDone, command *source.Command, args []json.RawMessage) (err error) {
	// If the command has a suggested fix function available, use it and apply
	// the edits to the workspace.
	if command.IsSuggestedFix() {
		return s.runSuggestedFixCommand(ctx, command, args)
	}
	switch command {
	case source.CommandTest:
		var uri protocol.DocumentURI
		var tests, benchmarks []string
		if err := source.UnmarshalArgs(args, &uri, &tests, &benchmarks); err != nil {
			return err
		}
		snapshot, _, ok, release, err := s.beginFileRequest(ctx, uri, source.UnknownKind)
		defer release()
		if !ok {
			return err
		}
		return s.runTests(ctx, snapshot, uri, work, tests, benchmarks)
	case source.CommandGenerate:
		var uri protocol.DocumentURI
		var recursive bool
		if err := source.UnmarshalArgs(args, &uri, &recursive); err != nil {
			return err
		}
		snapshot, _, ok, release, err := s.beginFileRequest(ctx, uri, source.UnknownKind)
		defer release()
		if !ok {
			return err
		}
		return s.runGoGenerate(ctx, snapshot, uri.SpanURI(), recursive, work)
	case source.CommandRegenerateCgo:
		var uri protocol.DocumentURI
		if err := source.UnmarshalArgs(args, &uri); err != nil {
			return err
		}
		mod := source.FileModification{
			URI:    uri.SpanURI(),
			Action: source.InvalidateMetadata,
		}
		return s.didModifyFiles(ctx, []source.FileModification{mod}, FromRegenerateCgo)
	case source.CommandTidy, source.CommandVendor:
		var uri protocol.DocumentURI
		if err := source.UnmarshalArgs(args, &uri); err != nil {
			return err
		}
		snapshot, _, ok, release, err := s.beginFileRequest(ctx, uri, source.UnknownKind)
		defer release()
		if !ok {
			return err
		}
		// The flow for `go mod tidy` and `go mod vendor` is almost identical,
		// so we combine them into one case for convenience.
		action := "tidy"
		if command == source.CommandVendor {
			action = "vendor"
		}
		return runSimpleGoCommand(ctx, snapshot, source.UpdateUserModFile|source.AllowNetwork, uri.SpanURI(), "mod", []string{action})
	case source.CommandUpdateGoSum:
		var uri protocol.DocumentURI
		if err := source.UnmarshalArgs(args, &uri); err != nil {
			return err
		}
		snapshot, _, ok, release, err := s.beginFileRequest(ctx, uri, source.UnknownKind)
		defer release()
		if !ok {
			return err
		}
		return runSimpleGoCommand(ctx, snapshot, source.UpdateUserModFile|source.AllowNetwork, uri.SpanURI(), "list", []string{"all"})
	case source.CommandCheckUpgrades:
		var uri protocol.DocumentURI
		var modules []string
		if err := source.UnmarshalArgs(args, &uri, &modules); err != nil {
			return err
		}
		snapshot, _, ok, release, err := s.beginFileRequest(ctx, uri, source.UnknownKind)
		defer release()
		if !ok {
			return err
		}
		upgrades, err := s.getUpgrades(ctx, snapshot, uri.SpanURI(), modules)
		if err != nil {
			return err
		}
		snapshot.View().RegisterModuleUpgrades(upgrades)
		// Re-diagnose the snapshot to publish the new module diagnostics.
		s.diagnoseSnapshot(snapshot, nil, false)
		return nil
	case source.CommandAddDependency, source.CommandUpgradeDependency:
		var uri protocol.DocumentURI
		var goCmdArgs []string
		var addRequire bool
		if err := source.UnmarshalArgs(args, &uri, &addRequire, &goCmdArgs); err != nil {
			return err
		}
		snapshot, _, ok, release, err := s.beginFileRequest(ctx, uri, source.UnknownKind)
		defer release()
		if !ok {
			return err
		}
		return s.runGoGetModule(ctx, snapshot, uri.SpanURI(), addRequire, goCmdArgs)
	case source.CommandRemoveDependency:
		var uri protocol.DocumentURI
		var modulePath string
		var onlyError bool
		if err := source.UnmarshalArgs(args, &uri, &onlyError, &modulePath); err != nil {
			return err
		}
		snapshot, fh, ok, release, err := s.beginFileRequest(ctx, uri, source.UnknownKind)
		defer release()
		if !ok {
			return err
		}
		// If the module is tidied apart from the one unused diagnostic, we can
		// run `go get module@none`, and then run `go mod tidy`. Otherwise, we
		// must make textual edits.
		// TODO(rstambler): In Go 1.17+, we will be able to use the go command
		// without checking if the module is tidy.
		if onlyError {
			if err := s.runGoGetModule(ctx, snapshot, uri.SpanURI(), false, []string{modulePath + "@none"}); err != nil {
				return err
			}
			return runSimpleGoCommand(ctx, snapshot, source.UpdateUserModFile|source.AllowNetwork, uri.SpanURI(), "mod", []string{"tidy"})
		}
		pm, err := snapshot.ParseMod(ctx, fh)
		if err != nil {
			return err
		}
		edits, err := dropDependency(snapshot, pm, modulePath)
		if err != nil {
			return err
		}
		response, err := s.client.ApplyEdit(ctx, &protocol.ApplyWorkspaceEditParams{
			Edit: protocol.WorkspaceEdit{
				DocumentChanges: []protocol.TextDocumentEdit{{
					TextDocument: protocol.VersionedTextDocumentIdentifier{
						Version: fh.Version(),
						TextDocumentIdentifier: protocol.TextDocumentIdentifier{
							URI: protocol.URIFromSpanURI(fh.URI()),
						},
					},
					Edits: edits,
				}},
			},
		})
		if err != nil {
			return err
		}
		if !response.Applied {
			return fmt.Errorf("edits not applied because of %s", response.FailureReason)
		}
	case source.CommandGoGetPackage:
		var uri protocol.DocumentURI
		var pkg string
		if err := source.UnmarshalArgs(args, &uri, &pkg); err != nil {
			return err
		}
		snapshot, _, ok, release, err := s.beginFileRequest(ctx, uri, source.UnknownKind)
		defer release()
		if !ok {
			return err
		}
		return s.runGoGetPackage(ctx, snapshot, uri.SpanURI(), pkg)

	case source.CommandToggleDetails:
		var fileURI protocol.DocumentURI
		if err := source.UnmarshalArgs(args, &fileURI); err != nil {
			return err
		}
		pkgDir := span.URIFromPath(filepath.Dir(fileURI.SpanURI().Filename()))
		s.gcOptimizationDetailsMu.Lock()
		if _, ok := s.gcOptimizationDetails[pkgDir]; ok {
			delete(s.gcOptimizationDetails, pkgDir)
			s.clearDiagnosticSource(gcDetailsSource)
		} else {
			s.gcOptimizationDetails[pkgDir] = struct{}{}
		}
		s.gcOptimizationDetailsMu.Unlock()
		// need to recompute diagnostics.
		// so find the snapshot
		snapshot, _, ok, release, err := s.beginFileRequest(ctx, fileURI, source.UnknownKind)
		defer release()
		if !ok {
			return err
		}
		s.diagnoseSnapshot(snapshot, nil, false)
	case source.CommandGenerateGoplsMod:
		var v source.View
		if len(args) == 0 {
			views := s.session.Views()
			if len(views) != 1 {
				return fmt.Errorf("cannot resolve view: have %d views", len(views))
			}
			v = views[0]
		} else {
			var uri protocol.DocumentURI
			if err := source.UnmarshalArgs(args, &uri); err != nil {
				return err
			}
			var err error
			v, err = s.session.ViewOf(uri.SpanURI())
			if err != nil {
				return err
			}
		}
		snapshot, release := v.Snapshot(ctx)
		defer release()
		modFile, err := cache.BuildGoplsMod(ctx, v.Folder(), snapshot)
		if err != nil {
			return errors.Errorf("getting workspace mod file: %w", err)
		}
		content, err := modFile.Format()
		if err != nil {
			return errors.Errorf("formatting mod file: %w", err)
		}
		filename := filepath.Join(v.Folder().Filename(), "gopls.mod")
		if err := ioutil.WriteFile(filename, content, 0644); err != nil {
			return errors.Errorf("writing mod file: %w", err)
		}
	default:
		return fmt.Errorf("unsupported command: %s", command.ID())
	}
	return nil
}

// dropDependency returns the edits to remove the given require from the go.mod
// file.
func dropDependency(snapshot source.Snapshot, pm *source.ParsedModule, modulePath string) ([]protocol.TextEdit, error) {
	// We need a private copy of the parsed go.mod file, since we're going to
	// modify it.
	copied, err := modfile.Parse("", pm.Mapper.Content, nil)
	if err != nil {
		return nil, err
	}
	if err := copied.DropRequire(modulePath); err != nil {
		return nil, err
	}
	copied.Cleanup()
	newContent, err := copied.Format()
	if err != nil {
		return nil, err
	}
	// Calculate the edits to be made due to the change.
	diff, err := snapshot.View().Options().ComputeEdits(pm.URI, string(pm.Mapper.Content), string(newContent))
	if err != nil {
		return nil, err
	}
	return source.ToProtocolEdits(pm.Mapper, diff)
}

func (s *Server) runTests(ctx context.Context, snapshot source.Snapshot, uri protocol.DocumentURI, work *workDone, tests, benchmarks []string) error {
	pkgs, err := snapshot.PackagesForFile(ctx, uri.SpanURI(), source.TypecheckWorkspace)
	if err != nil {
		return err
	}
	if len(pkgs) == 0 {
		return fmt.Errorf("package could not be found for file: %s", uri.SpanURI().Filename())
	}
	pkgPath := pkgs[0].ForTest()

	// create output
	buf := &bytes.Buffer{}
	ew := &eventWriter{ctx: ctx, operation: "test"}
	out := io.MultiWriter(ew, workDoneWriter{work}, buf)

	// Run `go test -run Func` on each test.
	var failedTests int
	for _, funcName := range tests {
		inv := &gocommand.Invocation{
			Verb:       "test",
			Args:       []string{pkgPath, "-v", "-count=1", "-run", fmt.Sprintf("^%s$", funcName)},
			WorkingDir: filepath.Dir(uri.SpanURI().Filename()),
		}
		if err := snapshot.RunGoCommandPiped(ctx, source.Normal, inv, out, out); err != nil {
			if errors.Is(err, context.Canceled) {
				return err
			}
			failedTests++
		}
	}

	// Run `go test -run=^$ -bench Func` on each test.
	var failedBenchmarks int
	for _, funcName := range benchmarks {
		inv := &gocommand.Invocation{
			Verb:       "test",
			Args:       []string{pkgPath, "-v", "-run=^$", "-bench", fmt.Sprintf("^%s$", funcName)},
			WorkingDir: filepath.Dir(uri.SpanURI().Filename()),
		}
		if err := snapshot.RunGoCommandPiped(ctx, source.Normal, inv, out, out); err != nil {
			if errors.Is(err, context.Canceled) {
				return err
			}
			failedBenchmarks++
		}
	}

	var title string
	if len(tests) > 0 && len(benchmarks) > 0 {
		title = "tests and benchmarks"
	} else if len(tests) > 0 {
		title = "tests"
	} else if len(benchmarks) > 0 {
		title = "benchmarks"
	} else {
		return errors.New("No functions were provided")
	}
	message := fmt.Sprintf("all %s passed", title)
	if failedTests > 0 && failedBenchmarks > 0 {
		message = fmt.Sprintf("%d / %d tests failed and %d / %d benchmarks failed", failedTests, len(tests), failedBenchmarks, len(benchmarks))
	} else if failedTests > 0 {
		message = fmt.Sprintf("%d / %d tests failed", failedTests, len(tests))
	} else if failedBenchmarks > 0 {
		message = fmt.Sprintf("%d / %d benchmarks failed", failedBenchmarks, len(benchmarks))
	}
	if failedTests > 0 || failedBenchmarks > 0 {
		message += "\n" + buf.String()
	}

	return s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
		Type:    protocol.Info,
		Message: message,
	})
}

func (s *Server) runGoGenerate(ctx context.Context, snapshot source.Snapshot, dir span.URI, recursive bool, work *workDone) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	er := &eventWriter{ctx: ctx, operation: "generate"}

	pattern := "."
	if recursive {
		pattern = "./..."
	}

	inv := &gocommand.Invocation{
		Verb:       "generate",
		Args:       []string{"-x", pattern},
		WorkingDir: dir.Filename(),
	}
	stderr := io.MultiWriter(er, workDoneWriter{work})
	if err := snapshot.RunGoCommandPiped(ctx, source.Normal, inv, er, stderr); err != nil {
		return err
	}
	return nil
}

func (s *Server) runGoGetPackage(ctx context.Context, snapshot source.Snapshot, uri span.URI, pkg string) error {
	stdout, err := snapshot.RunGoCommandDirect(ctx, source.WriteTemporaryModFile|source.AllowNetwork, &gocommand.Invocation{
		Verb:       "list",
		Args:       []string{"-f", "{{.Module.Path}}@{{.Module.Version}}", pkg},
		WorkingDir: filepath.Dir(uri.Filename()),
	})
	if err != nil {
		return err
	}
	ver := strings.TrimSpace(stdout.String())
	return s.runGoGetModule(ctx, snapshot, uri, true, []string{ver})
}

func (s *Server) runGoGetModule(ctx context.Context, snapshot source.Snapshot, uri span.URI, addRequire bool, args []string) error {
	if addRequire {
		// Using go get to create a new dependency results in an
		// `// indirect` comment we may not want. The only way to avoid it
		// is to add the require as direct first. Then we can use go get to
		// update go.sum and tidy up.
		if err := runSimpleGoCommand(ctx, snapshot, source.UpdateUserModFile, uri, "mod", append([]string{"edit", "-require"}, args...)); err != nil {
			return err
		}
	}
	return runSimpleGoCommand(ctx, snapshot, source.UpdateUserModFile|source.AllowNetwork, uri, "get", append([]string{"-d"}, args...))
}

func runSimpleGoCommand(ctx context.Context, snapshot source.Snapshot, mode source.InvocationFlags, uri span.URI, verb string, args []string) error {
	_, err := snapshot.RunGoCommandDirect(ctx, mode, &gocommand.Invocation{
		Verb:       verb,
		Args:       args,
		WorkingDir: filepath.Dir(uri.Filename()),
	})
	return err
}

func (s *Server) getUpgrades(ctx context.Context, snapshot source.Snapshot, uri span.URI, modules []string) (map[string]string, error) {
	stdout, err := snapshot.RunGoCommandDirect(ctx, source.Normal|source.AllowNetwork, &gocommand.Invocation{
		Verb:       "list",
		Args:       append([]string{"-m", "-u", "-json"}, modules...),
		WorkingDir: filepath.Dir(uri.Filename()),
	})
	if err != nil {
		return nil, err
	}

	upgrades := map[string]string{}
	for dec := json.NewDecoder(stdout); dec.More(); {
		mod := &gocommand.ModuleJSON{}
		if err := dec.Decode(mod); err != nil {
			return nil, err
		}
		if mod.Update == nil {
			continue
		}
		upgrades[mod.Path] = mod.Update.Version
	}
	return upgrades, nil
}
