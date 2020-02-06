package lsp

import (
	"context"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

func (s *Server) executeCommand(ctx context.Context, params *protocol.ExecuteCommandParams) (interface{}, error) {
	switch params.Command {
	case "tidy":
		if len(params.Arguments) == 0 || len(params.Arguments) > 1 {
			return nil, errors.Errorf("expected one file URI for call to `go mod tidy`, got %v", params.Arguments)
		}
		// Confirm that this action is being taken on a go.mod file.
		uri := span.NewURI(params.Arguments[0].(string))
		view, err := s.session.ViewOf(uri)
		if err != nil {
			return nil, err
		}
		snapshot := view.Snapshot()
		fh, err := snapshot.GetFile(uri)
		if err != nil {
			return nil, err
		}
		if fh.Identity().Kind != source.Mod {
			return nil, errors.Errorf("%s is not a mod file", uri)
		}
		// Run go.mod tidy on the view.
		if _, err := source.InvokeGo(ctx, view.Folder().Filename(), snapshot.Config(ctx).Env, "mod", "tidy"); err != nil {
			return nil, err
		}
	case "upgrade.dependency":
		if len(params.Arguments) < 2 {
			return nil, errors.Errorf("expected one file URI and one dependency for call to `go get`, got %v", params.Arguments)
		}
		uri := span.NewURI(params.Arguments[0].(string))
		view, err := s.session.ViewOf(uri)
		if err != nil {
			return nil, err
		}
		dep := params.Arguments[1].(string)
		// Run "go get" on the dependency to upgrade it to the latest version.
		if _, err := source.InvokeGo(ctx, view.Folder().Filename(), view.Snapshot().Config(ctx).Env, "get", dep); err != nil {
			return nil, err
		}
	}
	return nil, nil
}
