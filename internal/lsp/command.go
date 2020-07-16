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
	var found bool
	for _, command := range s.session.Options().SupportedCommands {
		if command == params.Command {
			found = true
			break
		}
	}
	if !found {
		return nil, fmt.Errorf("unsupported command detected: %s", params.Command)
	}
	switch params.Command {
	case source.CommandTest:
		unsaved := false
		for _, overlay := range s.session.Overlays() {
			if !overlay.Saved() {
				unsaved = true
				break
			}
		}
		if unsaved {
			return nil, s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
				Type:    protocol.Error,
				Message: "could not run tests, there are unsaved files in the view",
			})
		}
		funcName, uri, err := getRunTestArguments(params.Arguments)
		if err != nil {
			return nil, err
		}
		view, err := s.session.ViewOf(uri)
		if err != nil {
			return nil, err
		}
		go s.runTest(ctx, view.Snapshot(), funcName)
	case source.CommandGenerate:
		dir, recursive, err := getGenerateRequest(params.Arguments)
		if err != nil {
			return nil, err
		}
		go s.runGoGenerate(xcontext.Detach(ctx), dir, recursive)
	case source.CommandRegenerateCgo:
		mod := source.FileModification{
			URI:    protocol.DocumentURI(params.Arguments[0].(string)).SpanURI(),
			Action: source.InvalidateMetadata,
		}
		_, err := s.didModifyFiles(ctx, []source.FileModification{mod}, FromRegenerateCgo)
		return nil, err
	case source.CommandTidy, source.CommandVendor:
		if len(params.Arguments) == 0 || len(params.Arguments) > 1 {
			return nil, errors.Errorf("expected 1 argument, got %v", params.Arguments)
		}
		uri := protocol.DocumentURI(params.Arguments[0].(string))

		// The flow for `go mod tidy` and `go mod vendor` is almost identical,
		// so we combine them into one case for convenience.
		arg := "tidy"
		if params.Command == source.CommandVendor {
			arg = "vendor"
		}
		err := s.directGoModCommand(ctx, uri, "mod", []string{arg}...)
		return nil, err
	case source.CommandUpgradeDependency:
		if len(params.Arguments) < 2 {
			return nil, errors.Errorf("expected 2 arguments, got %v", params.Arguments)
		}
		uri := protocol.DocumentURI(params.Arguments[0].(string))
		deps := params.Arguments[1].(string)
		err := s.directGoModCommand(ctx, uri, "get", strings.Split(deps, " ")...)
		return nil, err
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

func (s *Server) runTest(ctx context.Context, snapshot source.Snapshot, funcName string) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	ew := &eventWriter{ctx: ctx, operation: "test"}
	msg := fmt.Sprintf("testing %s", funcName)
	wc := s.newProgressWriter(ctx, "test", msg, msg, cancel)
	defer wc.Close()

	messageType := protocol.Info
	message := "test passed"
	stderr := io.MultiWriter(ew, wc)

	if err := snapshot.RunGoCommandPiped(ctx, "test", []string{"-run", funcName}, ew, stderr); err != nil {
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

func (s *Server) runGoGenerate(ctx context.Context, dir string, recursive bool) error {
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
	uri := span.URIFromPath(dir)
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

func getRunTestArguments(args []interface{}) (string, span.URI, error) {
	if len(args) != 2 {
		return "", "", errors.Errorf("expected one test func name and one file path, got %v", args)
	}
	funcName, ok := args[0].(string)
	if !ok {
		return "", "", errors.Errorf("expected func name to be a string, got %T", args[0])
	}
	filename, ok := args[1].(string)
	if !ok {
		return "", "", errors.Errorf("expected file to be a string, got %T", args[1])
	}
	return funcName, span.URIFromPath(filename), nil
}

func getGenerateRequest(args []interface{}) (string, bool, error) {
	if len(args) != 2 {
		return "", false, errors.Errorf("expected exactly 2 arguments but got %d", len(args))
	}
	dir, ok := args[0].(string)
	if !ok {
		return "", false, errors.Errorf("expected dir to be a string value but got %T", args[0])
	}
	recursive, ok := args[1].(bool)
	if !ok {
		return "", false, errors.Errorf("expected recursive to be a boolean but got %T", args[1])
	}
	return dir, recursive, nil
}
