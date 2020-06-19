package mod

import (
	"context"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func Format(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) ([]protocol.TextEdit, error) {
	ctx, done := event.Start(ctx, "mod.Format")
	defer done()

	pmh, err := snapshot.ParseModHandle(ctx, fh)
	if err != nil {
		return nil, err
	}
	file, m, _, err := pmh.Parse(ctx)
	if err != nil {
		return nil, err
	}
	formatted, err := file.Format()
	if err != nil {
		return nil, err
	}
	// Calculate the edits to be made due to the change.
	diff := snapshot.View().Options().ComputeEdits(fh.URI(), string(m.Content), string(formatted))
	return source.ToProtocolEdits(m, diff)
}
