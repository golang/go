package source

import (
	"context"
)

func ModTidy(ctx context.Context, view View) error {
	cfg := view.Config(ctx)

	// Running `go mod tidy` modifies the file on disk directly.
	// Ideally, we should return modules that could possibly be removed
	// and apply each action as an edit.
	//
	// TODO(rstambler): This will be possible when golang/go#27005 is resolved.
	_, err := InvokeGo(ctx, view.Folder().Filename(), cfg.Env, "mod", "tidy")
	return err
}
