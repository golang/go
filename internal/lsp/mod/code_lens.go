package mod

import (
	"context"
	"fmt"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/trace"
)

func CodeLens(ctx context.Context, snapshot source.Snapshot, uri span.URI) ([]protocol.CodeLens, error) {
	realURI, _ := snapshot.View().ModFiles()
	if realURI == "" {
		return nil, nil
	}
	// Only get code lens on the go.mod for the view.
	if uri != realURI {
		return nil, nil
	}
	ctx, done := trace.StartSpan(ctx, "mod.CodeLens", telemetry.File.Of(realURI))
	defer done()

	fh, err := snapshot.GetFile(realURI)
	if err != nil {
		return nil, err
	}
	f, m, upgrades, err := snapshot.ModHandle(ctx, fh).Upgrades(ctx)
	if err != nil {
		return nil, err
	}
	var codelens []protocol.CodeLens
	var allUpgrades []string
	for _, req := range f.Require {
		dep := req.Mod.Path
		latest, ok := upgrades[dep]
		if !ok {
			continue
		}
		// Get the range of the require directive.
		rng, err := positionsToRange(uri, m, req.Syntax.Start, req.Syntax.End)
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
		allUpgrades = append(allUpgrades, dep)
	}
	// If there is at least 1 upgrade, add an "Upgrade all dependencies" to the module statement.
	if module := f.Module; len(allUpgrades) > 0 && module != nil && module.Syntax != nil {
		// Get the range of the module directive.
		rng, err := positionsToRange(uri, m, module.Syntax.Start, module.Syntax.End)
		if err != nil {
			return nil, err
		}
		codelens = append(codelens, protocol.CodeLens{
			Range: rng,
			Command: protocol.Command{
				Title:     "Upgrade all dependencies",
				Command:   "upgrade.dependency",
				Arguments: []interface{}{uri, strings.Join(append([]string{"-u"}, allUpgrades...), " ")},
			},
		})
	}
	return codelens, err
}

func positionsToRange(uri span.URI, m *protocol.ColumnMapper, s, e modfile.Position) (protocol.Range, error) {
	line, col, err := m.Converter.ToPosition(s.Byte)
	if err != nil {
		return protocol.Range{}, err
	}
	start := span.NewPoint(line, col, s.Byte)
	line, col, err = m.Converter.ToPosition(e.Byte)
	if err != nil {
		return protocol.Range{}, err
	}
	end := span.NewPoint(line, col, e.Byte)
	rng, err := m.Range(span.New(uri, start, end))
	if err != nil {
		return protocol.Range{}, err
	}
	return rng, err
}
