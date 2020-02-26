package lsp

import (
	"context"
	"strings"

	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	errors "golang.org/x/xerrors"
)

func (s *Server) executeCommand(ctx context.Context, params *protocol.ExecuteCommandParams) (interface{}, error) {
	switch params.Command {
	case "tidy":
		if len(params.Arguments) == 0 || len(params.Arguments) > 1 {
			return nil, errors.Errorf("expected one file URI for call to `go mod tidy`, got %v", params.Arguments)
		}
		uri := protocol.DocumentURI(params.Arguments[0].(string))
		snapshot, _, ok, err := s.beginFileRequest(uri, source.Mod)
		if !ok {
			return nil, err
		}
		// Run go.mod tidy on the view.
		inv := gocommand.Invocation{
			Verb:       "mod",
			Args:       []string{"tidy"},
			Env:        snapshot.Config(ctx).Env,
			WorkingDir: snapshot.View().Folder().Filename(),
		}
		if _, err := inv.Run(ctx); err != nil {
			return nil, err
		}
	case "upgrade.dependency":
		if len(params.Arguments) < 2 {
			return nil, errors.Errorf("expected one file URI and one dependency for call to `go get`, got %v", params.Arguments)
		}
		uri := protocol.DocumentURI(params.Arguments[0].(string))
		deps := params.Arguments[1].(string)
		snapshot, _, ok, err := s.beginFileRequest(uri, source.UnknownKind)
		if !ok {
			return nil, err
		}
		// Run "go get" on the dependency to upgrade it to the latest version.
		inv := gocommand.Invocation{
			Verb:       "get",
			Args:       strings.Split(deps, " "),
			Env:        snapshot.Config(ctx).Env,
			WorkingDir: snapshot.View().Folder().Filename(),
		}
		if _, err := inv.Run(ctx); err != nil {
			return nil, err
		}
	}
	return nil, nil
}
