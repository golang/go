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
	"golang.org/x/xerrors"
	errors "golang.org/x/xerrors"
)

func (s *Server) executeCommand(ctx context.Context, params *protocol.ExecuteCommandParams) (interface{}, error) {
	switch params.Command {
	case source.CommandTest:
		if len(s.session.UnsavedFiles()) != 0 {
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
	case source.CommandTidy:
		if len(params.Arguments) == 0 || len(params.Arguments) > 1 {
			return nil, errors.Errorf("expected one file URI for call to `go mod tidy`, got %v", params.Arguments)
		}
		uri := protocol.DocumentURI(params.Arguments[0].(string))
		snapshot, _, ok, err := s.beginFileRequest(uri, source.Mod)
		if !ok {
			return nil, err
		}
		cfg := snapshot.Config(ctx)
		// Run go.mod tidy on the view.
		inv := gocommand.Invocation{
			Verb:       "mod",
			Args:       []string{"tidy"},
			Env:        cfg.Env,
			WorkingDir: snapshot.View().Folder().Filename(),
		}
		gocmdRunner := packagesinternal.GetGoCmdRunner(cfg)
		if _, err := gocmdRunner.Run(ctx, inv); err != nil {
			return nil, err
		}
	case source.CommandUpgradeDependency:
		if len(params.Arguments) < 2 {
			return nil, errors.Errorf("expected one file URI and one dependency for call to `go get`, got %v", params.Arguments)
		}
		uri := protocol.DocumentURI(params.Arguments[0].(string))
		deps := params.Arguments[1].(string)
		snapshot, _, ok, err := s.beginFileRequest(uri, source.UnknownKind)
		if !ok {
			return nil, err
		}
		cfg := snapshot.Config(ctx)
		// Run "go get" on the dependency to upgrade it to the latest version.
		inv := gocommand.Invocation{
			Verb:       "get",
			Args:       strings.Split(deps, " "),
			Env:        cfg.Env,
			WorkingDir: snapshot.View().Folder().Filename(),
		}
		gocmdRunner := packagesinternal.GetGoCmdRunner(cfg)
		if _, err := gocmdRunner.Run(ctx, inv); err != nil {
			return nil, err
		}
	}
	return nil, nil
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
		if !xerrors.Is(err, context.Canceled) {
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
