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
		fh, err := view.Snapshot().GetFile(uri)
		if err != nil {
			return nil, err
		}
		if fh.Identity().Kind != source.Mod {
			return nil, errors.Errorf("%s is not a mod file", uri)
		}
		// Run go.mod tidy on the view.
		// TODO: This should go through the ModTidyHandle on the view.
		// That will also allow us to move source.InvokeGo into internal/lsp/cache.
		if _, err := source.InvokeGo(ctx, view.Folder().Filename(), view.Config(ctx).Env, "mod", "tidy"); err != nil {
			return nil, err
		}
	}
	return nil, nil
}
