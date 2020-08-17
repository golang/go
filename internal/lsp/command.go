// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"io"
	"path"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/xcontext"
	errors "golang.org/x/xerrors"
)

func (s *Server) executeCommand(ctx context.Context, params *protocol.ExecuteCommandParams) (interface{}, error) {
	var command *source.Command
	for _, c := range source.Commands {
		if c.Name == params.Command {
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
		return nil, fmt.Errorf("%s is not a supported command", command.Name)
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
		case source.CommandTest.Name, source.CommandGenerate.Name, source.CommandToggleDetails.Name:
			// TODO(PJW): for Toggle, not an error if it is being disabled
			err := fmt.Errorf("cannot run command %s: unsaved files in the view", params.Command)
			s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
				Type:    protocol.Error,
				Message: err.Error(),
			})
			return nil, err
		}
	}
	// If the command has a suggested fix function available, use it and apply
	// the edits to the workspace.
	if command.IsSuggestedFix() {
		var uri protocol.DocumentURI
		var rng protocol.Range
		if err := source.UnmarshalArgs(params.Arguments, &uri, &rng); err != nil {
			return nil, err
		}
		snapshot, fh, ok, release, err := s.beginFileRequest(ctx, uri, source.Go)
		defer release()
		if !ok {
			return nil, err
		}
		edits, err := command.SuggestedFix(ctx, snapshot, fh, rng)
		if err != nil {
			return nil, err
		}
		r, err := s.client.ApplyEdit(ctx, &protocol.ApplyWorkspaceEditParams{
			Edit: protocol.WorkspaceEdit{
				DocumentChanges: edits,
			},
		})
		if err != nil {
			return nil, err
		}
		if !r.Applied {
			return nil, s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
				Type:    protocol.Error,
				Message: fmt.Sprintf("%s failed: %v", params.Command, r.FailureReason),
			})
		}
		return nil, nil
	}
	// Default commands that don't have suggested fix functions.
	switch command {
	case source.CommandTest:
		var uri protocol.DocumentURI
		var tests, benchmarks []string
		if err := source.UnmarshalArgs(params.Arguments, &uri, &tests, &benchmarks); err != nil {
			return nil, err
		}
		snapshot, _, ok, release, err := s.beginFileRequest(ctx, uri, source.UnknownKind)
		defer release()
		if !ok {
			return nil, err
		}
		go s.runTests(ctx, snapshot, uri, params.WorkDoneToken, tests, benchmarks)
	case source.CommandGenerate:
		var uri protocol.DocumentURI
		var recursive bool
		if err := source.UnmarshalArgs(params.Arguments, &uri, &recursive); err != nil {
			return nil, err
		}
		snapshot, _, ok, release, err := s.beginFileRequest(ctx, uri, source.UnknownKind)
		defer release()
		if !ok {
			return nil, err
		}
		go s.runGoGenerate(xcontext.Detach(ctx), snapshot, uri.SpanURI(), recursive, params.WorkDoneToken)
	case source.CommandRegenerateCgo:
		var uri protocol.DocumentURI
		if err := source.UnmarshalArgs(params.Arguments, &uri); err != nil {
			return nil, err
		}
		mod := source.FileModification{
			URI:    uri.SpanURI(),
			Action: source.InvalidateMetadata,
		}
		err := s.didModifyFiles(ctx, []source.FileModification{mod}, FromRegenerateCgo)
		return nil, err
	case source.CommandTidy, source.CommandVendor:
		var uri protocol.DocumentURI
		if err := source.UnmarshalArgs(params.Arguments, &uri); err != nil {
			return nil, err
		}
		// The flow for `go mod tidy` and `go mod vendor` is almost identical,
		// so we combine them into one case for convenience.
		a := "tidy"
		if command == source.CommandVendor {
			a = "vendor"
		}
		err := s.directGoModCommand(ctx, uri, "mod", []string{a}...)
		return nil, err
	case source.CommandUpgradeDependency:
		var uri protocol.DocumentURI
		var goCmdArgs []string
		if err := source.UnmarshalArgs(params.Arguments, &uri, &goCmdArgs); err != nil {
			return nil, err
		}
		err := s.directGoModCommand(ctx, uri, "get", goCmdArgs...)
		return nil, err
	case source.CommandToggleDetails:
		var fileURI span.URI
		if err := source.UnmarshalArgs(params.Arguments, &fileURI); err != nil {
			return nil, err
		}
		pkgDir := span.URIFromPath(path.Dir(fileURI.Filename()))
		s.gcOptimizationDetailsMu.Lock()
		if _, ok := s.gcOptimizatonDetails[pkgDir]; ok {
			delete(s.gcOptimizatonDetails, pkgDir)
		} else {
			s.gcOptimizatonDetails[pkgDir] = struct{}{}
		}
		s.gcOptimizationDetailsMu.Unlock()
		event.Log(ctx, fmt.Sprintf("gc_details %s now %v %v", pkgDir, s.gcOptimizatonDetails[pkgDir],
			s.gcOptimizatonDetails))
		// need to recompute diagnostics.
		// so find the snapshot
		sv, err := s.session.ViewOf(fileURI)
		if err != nil {
			return nil, err
		}
		snapshot, release := sv.Snapshot(ctx)
		defer release()
		s.diagnoseSnapshot(snapshot)
		return nil, nil
	default:
		return nil, fmt.Errorf("unknown command: %s", params.Command)
	}
	return nil, nil
}

func (s *Server) directGoModCommand(ctx context.Context, uri protocol.DocumentURI, verb string, args ...string) error {
	view, err := s.session.ViewOf(uri.SpanURI())
	if err != nil {
		return err
	}
	snapshot, release := view.Snapshot(ctx)
	defer release()
	return snapshot.RunGoCommandDirect(ctx, verb, args)
}

func (s *Server) runTests(ctx context.Context, snapshot source.Snapshot, uri protocol.DocumentURI, token protocol.ProgressToken, tests, benchmarks []string) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	pkgs, err := snapshot.PackagesForFile(ctx, uri.SpanURI(), source.TypecheckWorkspace)
	if err != nil {
		return err
	}
	if len(pkgs) == 0 {
		return fmt.Errorf("package could not be found for file: %s", uri.SpanURI().Filename())
	}
	pkgPath := pkgs[0].PkgPath()

	// create output
	ew := &eventWriter{ctx: ctx, operation: "test"}
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
	msg := fmt.Sprintf("Running %s...", title)
	wc := s.progress.newWriter(ctx, title, msg, msg, token, cancel)
	defer wc.Close()

	stderr := io.MultiWriter(ew, wc)

	// run `go test -run Func` on each test
	var failedTests int
	for _, funcName := range tests {
		args := []string{pkgPath, "-run", fmt.Sprintf("^%s$", funcName)}
		if err := snapshot.RunGoCommandPiped(ctx, "test", args, ew, stderr); err != nil {
			if errors.Is(err, context.Canceled) {
				return err
			}
			failedTests++
		}
	}

	// run `go test -run=^$ -bench Func` on each test
	var failedBenchmarks int
	for _, funcName := range tests {
		args := []string{pkgPath, "-run=^$", "-bench", fmt.Sprintf("^%s$", funcName)}
		if err := snapshot.RunGoCommandPiped(ctx, "test", args, ew, stderr); err != nil {
			if errors.Is(err, context.Canceled) {
				return err
			}
			failedBenchmarks++
		}
	}

	messageType := protocol.Info
	message := fmt.Sprintf("all %s passed", title)
	if failedTests > 0 || failedBenchmarks > 0 {
		messageType = protocol.Error
	}
	if failedTests > 0 && failedBenchmarks > 0 {
		message = fmt.Sprintf("%d / %d tests failed and %d / %d benchmarks failed", failedTests, len(tests), failedBenchmarks, len(benchmarks))
	} else if failedTests > 0 {
		message = fmt.Sprintf("%d / %d tests failed", failedTests, len(tests))
	} else if failedBenchmarks > 0 {
		message = fmt.Sprintf("%d / %d benchmarks failed", failedBenchmarks, len(benchmarks))
	}

	return s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
		Type:    messageType,
		Message: message,
	})
}

// GenerateWorkDoneTitle is the title used in progress reporting for go
// generate commands. It is exported for testing purposes.
const GenerateWorkDoneTitle = "generate"

func (s *Server) runGoGenerate(ctx context.Context, snapshot source.Snapshot, uri span.URI, recursive bool, token protocol.ProgressToken) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	er := &eventWriter{ctx: ctx, operation: "generate"}
	wc := s.progress.newWriter(ctx, GenerateWorkDoneTitle, "running go generate", "started go generate, check logs for progress", token, cancel)
	defer wc.Close()
	args := []string{"-x"}
	if recursive {
		args = append(args, "./...")
	}

	stderr := io.MultiWriter(er, wc)

	if err := snapshot.RunGoCommandPiped(ctx, "generate", args, er, stderr); err != nil {
		if errors.Is(err, context.Canceled) {
			return nil
		}
		event.Error(ctx, "generate: command error", err, tag.Directory.Of(uri.Filename()))
		return s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
			Type:    protocol.Error,
			Message: "go generate exited with an error, check gopls logs",
		})
	}
	return nil
}
