package mod

import (
	"context"
	"fmt"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/trace"
)

func CodeLens(ctx context.Context, snapshot source.Snapshot, uri span.URI) ([]protocol.CodeLens, error) {
	realURI, _ := snapshot.View().ModFiles()
	// Check the case when the tempModfile flag is turned off.
	if realURI == "" {
		return nil, nil
	}
	// Only get code lens on the go.mod for the view.
	if uri != realURI {
		return nil, nil
	}
	ctx, done := trace.StartSpan(ctx, "mod.CodeLens", telemetry.File.Of(realURI))
	defer done()

	pmh, err := snapshot.ParseModHandle(ctx)
	if err != nil {
		return nil, err
	}
	f, m, upgrades, err := pmh.Upgrades(ctx)
	if err != nil {
		return nil, err
	}
	var codelens []protocol.CodeLens
	for _, req := range f.Require {
		dep := req.Mod.Path
		latest, ok := upgrades[dep]
		if !ok {
			continue
		}
		// Get the range of the require directive.
		s, e := req.Syntax.Start, req.Syntax.End
		line, col, err := m.Converter.ToPosition(s.Byte)
		if err != nil {
			return nil, err
		}
		start := span.NewPoint(line, col, s.Byte)
		line, col, err = m.Converter.ToPosition(e.Byte)
		if err != nil {
			return nil, err
		}
		end := span.NewPoint(line, col, e.Byte)
		rng, err := m.Range(span.New(uri, start, end))
		if err != nil {
			return nil, err
		}
		codelens = append(codelens, protocol.CodeLens{
			Range: rng,
			Command: protocol.Command{
				Title:     fmt.Sprintf("Upgrade dependency to %s", latest),
				Command:   "upgrade.dependency",
				Arguments: []interface{}{uri, dep},
			},
		})
	}
	return codelens, err
}
