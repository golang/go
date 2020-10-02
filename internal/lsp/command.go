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

	"golang.org/x/tools/internal/event"
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
		if command.Name == name {
			match = true
			break
		}
	}
	if !match {
		return nil, fmt.Errorf("%s is not a supported command", command.ID())
	}
	title := command.Title
	if title == "" {
		title = command.Name
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
		case source.CommandTest.ID(), source.CommandGenerate.ID(), source.CommandToggleDetails.ID():
			// TODO(PJW): for Toggle, not an error if it is being disabled
			err := errors.New("unsaved files in the view")
			s.showCommandError(ctx, title, err)
			return nil, err
		}
	}
	// If the command has a suggested fix function available, use it and apply
	// the edits to the workspace.
	if command.IsSuggestedFix() {
		err := s.runSuggestedFixCommand(ctx, command, params.Arguments)
		if err != nil {
			s.showCommandError(ctx, title, err)
		}
		return nil, err
	}
	ctx, cancel := context.WithCancel(xcontext.Detach(ctx))
	// Start progress prior to spinning off a goroutine specifically so that
	// clients are aware of the work item before the command completes. This
	// matters for regtests, where having a continuous thread of work is
	// convenient for assertions.
	work := s.progress.start(ctx, title, "Running...", params.WorkDoneToken, cancel)
	if command.Synchronous {
		return nil, s.runCommand(ctx, work, command, params.Arguments)
	}
	go func() {
		defer cancel()
		err := s.runCommand(ctx, work, command, params.Arguments)
		switch {
		case errors.Is(err, context.Canceled):
			work.end(title + ": canceled")
		case err != nil:
			event.Error(ctx, fmt.Sprintf("%s: command error", title), err)
			work.end(title + ": failed")
			// Show a message when work completes with error, because the progress end
			// message is typically dismissed immediately by LSP clients.
			s.showCommandError(ctx, title, err)
		default:
			work.end(command.ID() + ": completed")
		}
	}()
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

func (s *Server) runCommand(ctx context.Context, work *workDone, command *source.Command, args []json.RawMessage) error {
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
		// The flow for `go mod tidy` and `go mod vendor` is almost identical,
		// so we combine them into one case for convenience.
		a := "tidy"
		if command == source.CommandVendor {
			a = "vendor"
		}
		return s.directGoModCommand(ctx, uri, "mod", []string{a}...)
	case source.CommandUpgradeDependency:
		var uri protocol.DocumentURI
		var goCmdArgs []string
		if err := source.UnmarshalArgs(args, &uri, &goCmdArgs); err != nil {
			return err
		}
		return s.directGoModCommand(ctx, uri, "get", goCmdArgs...)
	case source.CommandToggleDetails:
		var fileURI span.URI
		if err := source.UnmarshalArgs(args, &fileURI); err != nil {
			return err
		}
		pkgDir := span.URIFromPath(filepath.Dir(fileURI.Filename()))
		s.gcOptimizationDetailsMu.Lock()
		if _, ok := s.gcOptimizatonDetails[pkgDir]; ok {
			delete(s.gcOptimizatonDetails, pkgDir)
		} else {
			s.gcOptimizatonDetails[pkgDir] = struct{}{}
		}
		s.gcOptimizationDetailsMu.Unlock()
		// need to recompute diagnostics.
		// so find the snapshot
		sv, err := s.session.ViewOf(fileURI)
		if err != nil {
			return err
		}
		snapshot, release := sv.Snapshot(ctx)
		defer release()
		s.diagnoseSnapshot(snapshot, nil)
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
		modFile, err := snapshot.BuildWorkspaceModFile(ctx)
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

func (s *Server) directGoModCommand(ctx context.Context, uri protocol.DocumentURI, verb string, args ...string) error {
	view, err := s.session.ViewOf(uri.SpanURI())
	if err != nil {
		return err
	}
	wdir := filepath.Dir(uri.SpanURI().Filename())
	snapshot, release := view.Snapshot(ctx)
	defer release()
	return snapshot.RunGoCommandDirect(ctx, wdir, verb, args)
}

func (s *Server) runTests(ctx context.Context, snapshot source.Snapshot, uri protocol.DocumentURI, work *workDone, tests, benchmarks []string) error {
	pkgs, err := snapshot.PackagesForFile(ctx, uri.SpanURI(), source.TypecheckWorkspace)
	if err != nil {
		return err
	}
	if len(pkgs) == 0 {
		return fmt.Errorf("package could not be found for file: %s", uri.SpanURI().Filename())
	}
	pkgPath := pkgs[0].PkgPath()

	// create output
	buf := &bytes.Buffer{}
	ew := &eventWriter{ctx: ctx, operation: "test"}
	out := io.MultiWriter(ew, workDoneWriter{work}, buf)

	wdir := filepath.Dir(uri.SpanURI().Filename())

	// Run `go test -run Func` on each test.
	var failedTests int
	for _, funcName := range tests {
		args := []string{pkgPath, "-v", "-count=1", "-run", fmt.Sprintf("^%s$", funcName)}
		if err := snapshot.RunGoCommandPiped(ctx, wdir, "test", args, out, out); err != nil {
			if errors.Is(err, context.Canceled) {
				return err
			}
			failedTests++
		}
	}

	// Run `go test -run=^$ -bench Func` on each test.
	var failedBenchmarks int
	for _, funcName := range benchmarks {
		args := []string{pkgPath, "-v", "-run=^$", "-bench", fmt.Sprintf("^%s$", funcName)}
		if err := snapshot.RunGoCommandPiped(ctx, wdir, "test", args, out, out); err != nil {
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
	messageType := protocol.Info
	if failedTests > 0 || failedBenchmarks > 0 {
		messageType = protocol.Error
		message += "\n" + buf.String()
	}

	return s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
		Type:    messageType,
		Message: message,
	})
}

func (s *Server) runGoGenerate(ctx context.Context, snapshot source.Snapshot, dir span.URI, recursive bool, work *workDone) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	er := &eventWriter{ctx: ctx, operation: "generate"}
	args := []string{"-x"}
	pattern := "."
	if recursive {
		pattern = "..."
	}
	args = append(args, pattern)

	stderr := io.MultiWriter(er, workDoneWriter{work})

	if err := snapshot.RunGoCommandPiped(ctx, dir.Filename(), "generate", args, er, stderr); err != nil {
		return err
	}
	return nil
}
