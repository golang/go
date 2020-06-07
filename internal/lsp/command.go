// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"io"
	"path/filepath"
	"strings"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/packagesinternal"
	"golang.org/x/tools/internal/xcontext"
	errors "golang.org/x/xerrors"
)

func (s *Server) executeCommand(ctx context.Context, params *protocol.ExecuteCommandParams) (interface{}, error) {
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
		snapshot, fh, ok, err := s.beginFileRequest(protocol.DocumentURI(uri), source.Go)
		if !ok {
			return nil, err
		}
		dir := filepath.Dir(fh.Identity().URI.Filename())
		go s.runTest(ctx, funcName, dir, snapshot)
	case source.CommandGenerate:
		dir, recursive, err := getGenerateRequest(params.Arguments)
		if err != nil {
			return nil, err
		}
		go s.runGenerate(xcontext.Detach(ctx), dir, recursive)
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
		err := s.goModCommand(ctx, uri, "mod", []string{arg}...)
		return nil, err
	case source.CommandUpgradeDependency:
		if len(params.Arguments) < 2 {
			return nil, errors.Errorf("expected 2 arguments, got %v", params.Arguments)
		}
		uri := protocol.DocumentURI(params.Arguments[0].(string))
		deps := params.Arguments[1].(string)
		err := s.goModCommand(ctx, uri, "get", strings.Split(deps, " ")...)
		return nil, err
	}
	return nil, nil
}

func (s *Server) goModCommand(ctx context.Context, uri protocol.DocumentURI, verb string, args ...string) error {
	view, err := s.session.ViewOf(uri.SpanURI())
	if err != nil {
		return err
	}
	snapshot := view.Snapshot()
	cfg := snapshot.Config(ctx)
	inv := gocommand.Invocation{
		Verb:       verb,
		Args:       args,
		Env:        cfg.Env,
		WorkingDir: view.Folder().Filename(),
	}
	gocmdRunner := packagesinternal.GetGoCmdRunner(cfg)
	_, err = gocmdRunner.Run(ctx, inv)
	return err
}

func (s *Server) runTest(ctx context.Context, funcName string, dir string, snapshot source.Snapshot) {
	args := []string{"-run", funcName, dir}
	inv := gocommand.Invocation{
		Verb:       "test",
		Args:       args,
		Env:        snapshot.Config(ctx).Env,
		WorkingDir: dir,
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	er := &eventWriter{ctx: ctx, operation: "test"}
	wc := s.newProgressWriter(ctx, "test", "running "+funcName, cancel)
	defer wc.Close()

	messageType := protocol.Info
	message := "test passed"
	stderr := io.MultiWriter(er, wc)
	if err := inv.RunPiped(ctx, er, stderr); err != nil {
		event.Error(ctx, "test: command error", err, tag.Directory.Of(dir))
		if !errors.Is(err, context.Canceled) {
			messageType = protocol.Error
			message = "test failed"
		}
	}

	s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
		Type:    messageType,
		Message: message,
	})
}

func getRunTestArguments(args []interface{}) (string, string, error) {
	if len(args) != 2 {
		return "", "", errors.Errorf("expected one test func name and one file path, got %v", args)
	}

	funcName, ok := args[0].(string)
	if !ok {
		return "", "", errors.Errorf("expected func name to be a string, got %T", args[0])
	}

	file, ok := args[1].(string)
	if !ok {
		return "", "", errors.Errorf("expected file to be a string, got %T", args[1])
	}

	return funcName, file, nil
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
