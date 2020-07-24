// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"io"
	"strings"

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
		switch command {
		case source.CommandTest, source.CommandGenerate:
			return nil, s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
				Type:    protocol.Error,
				Message: fmt.Sprintf("cannot run command %s: unsaved files in the view", params.Command),
			})
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
		snapshot, fh, ok, err := s.beginFileRequest(ctx, uri, source.Go)
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
		var flag string
		var funcName string
		if err := source.UnmarshalArgs(params.Arguments, &uri, &flag, &funcName); err != nil {
			return nil, err
		}
		snapshot, _, ok, err := s.beginFileRequest(ctx, uri, source.UnknownKind)
		if !ok {
			return nil, err
		}
		go s.runTest(ctx, snapshot, []string{flag, funcName})
	case source.CommandGenerate:
		var uri protocol.DocumentURI
		var recursive bool
		if err := source.UnmarshalArgs(params.Arguments, &uri, &recursive); err != nil {
			return nil, err
		}
		go s.runGoGenerate(xcontext.Detach(ctx), uri.SpanURI(), recursive)
	case source.CommandRegenerateCgo:
		var uri protocol.DocumentURI
		if err := source.UnmarshalArgs(params.Arguments, &uri); err != nil {
			return nil, err
		}
		mod := source.FileModification{
			URI:    uri.SpanURI(),
			Action: source.InvalidateMetadata,
		}
		_, err := s.didModifyFiles(ctx, []source.FileModification{mod}, FromRegenerateCgo)
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
		var deps []string
		if err := source.UnmarshalArgs(params.Arguments, &uri, &deps); err != nil {
			return nil, err
		}
		err := s.directGoModCommand(ctx, uri, "get", deps...)
		return nil, err
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
	return view.Snapshot().RunGoCommandDirect(ctx, verb, args)
}

func (s *Server) runTest(ctx context.Context, snapshot source.Snapshot, args []string) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	ew := &eventWriter{ctx: ctx, operation: "test"}
	msg := fmt.Sprintf("running `go test %s`", strings.Join(args, " "))
	wc := s.newProgressWriter(ctx, "test", msg, msg, cancel)
	defer wc.Close()

	messageType := protocol.Info
	message := "test passed"
	stderr := io.MultiWriter(ew, wc)

	if err := snapshot.RunGoCommandPiped(ctx, "test", args, ew, stderr); err != nil {
		if errors.Is(err, context.Canceled) {
			return err
		}
		messageType = protocol.Error
		message = "test failed"
	}
	return s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
		Type:    messageType,
		Message: message,
	})
}

// GenerateWorkDoneTitle is the title used in progress reporting for go
// generate commands. It is exported for testing purposes.
const GenerateWorkDoneTitle = "generate"

func (s *Server) runGoGenerate(ctx context.Context, uri span.URI, recursive bool) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	er := &eventWriter{ctx: ctx, operation: "generate"}
	wc := s.newProgressWriter(ctx, GenerateWorkDoneTitle, "running go generate", "started go generate, check logs for progress", cancel)
	defer wc.Close()
	args := []string{"-x"}
	if recursive {
		args = append(args, "./...")
	}

	stderr := io.MultiWriter(er, wc)
	view, err := s.session.ViewOf(uri)
	if err != nil {
		return err
	}
	snapshot := view.Snapshot()
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
