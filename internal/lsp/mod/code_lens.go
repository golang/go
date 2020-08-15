package mod

import (
	"context"
	"fmt"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// CodeLens computes code lens for a go.mod file.
func CodeLens(ctx context.Context, snapshot source.Snapshot, uri span.URI) ([]protocol.CodeLens, error) {
	if !snapshot.View().Options().EnabledCodeLens[source.CommandUpgradeDependency.Name] {
		return nil, nil
	}
	ctx, done := event.Start(ctx, "mod.CodeLens", tag.URI.Of(uri))
	defer done()

	// Only show go.mod code lenses in module mode, for the view's go.mod.
	if modURI := snapshot.View().ModFile(); modURI == "" || modURI != uri {
		return nil, nil
	}
	fh, err := snapshot.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	pm, err := snapshot.ParseMod(ctx, fh)
	if err != nil {
		return nil, err
	}
	upgrades, err := snapshot.ModUpgrade(ctx, fh)
	if err != nil {
		return nil, err
	}

	var (
		codelens    []protocol.CodeLens
		allUpgrades []string
	)

	for _, req := range pm.File.Require {
		dep := req.Mod.Path
		latest, ok := upgrades[dep]
		if !ok {
			continue
		}
		// Get the range of the require directive.
		rng, err := positionsToRange(uri, pm.Mapper, req.Syntax.Start, req.Syntax.End)
		if err != nil {
			return nil, err
		}
		upgradeDepArgs, err := source.MarshalArgs(uri, []string{dep})
		if err != nil {
			return nil, err
		}
		codelens = append(codelens, protocol.CodeLens{
			Range: rng,
			Command: protocol.Command{
				Title:     fmt.Sprintf("Upgrade dependency to %s", latest),
				Command:   source.CommandUpgradeDependency.Name,
				Arguments: upgradeDepArgs,
			},
		})
		allUpgrades = append(allUpgrades, dep)
	}

	module := pm.File.Module
	if module == nil || module.Syntax == nil {
		return codelens, nil
	}
	// Get the range of the module directive.
	rng, err := positionsToRange(uri, pm.Mapper, module.Syntax.Start, module.Syntax.End)
	if err != nil {
		return nil, err
	}

	// Add go mod vendor and go mod tidy lenses
	goModArgs, err := source.MarshalArgs(uri)
	if err != nil {
		return nil, err
	}
	codelens = append(codelens, protocol.CodeLens{
		Range: rng,
		Command: protocol.Command{
			Title:     "Sync vendor directory",
			Command:   source.CommandVendor.Name,
			Arguments: goModArgs,
		},
	})
	tidied, err := snapshot.ModTidy(ctx, fh)
	if err != nil && err != source.ErrTmpModfileUnsupported {
		return nil, err
	}
	if len(tidied.Errors) > 0 {
		codelens = append(codelens, protocol.CodeLens{
			Range: rng,
			Command: protocol.Command{
				Title:     "Remove unused dependencies",
				Command:   source.CommandTidy.Name,
				Arguments: goModArgs,
			},
		})
	}

	// If there is at least 1 upgrade, add an "Upgrade all dependencies" to the module statement.
	if len(allUpgrades) > 0 {
		upgradeDepArgs, err := source.MarshalArgs(uri, append([]string{"-u"}, allUpgrades...))
		if err != nil {
			return nil, err
		}
		codelens = append(codelens, protocol.CodeLens{
			Range: rng,
			Command: protocol.Command{
				Title:     "Upgrade all dependencies",
				Command:   source.CommandUpgradeDependency.Name,
				Arguments: upgradeDepArgs,
			},
		})
	}
	return codelens, nil
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
