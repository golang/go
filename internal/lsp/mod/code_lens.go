package mod

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// LensFuncs returns the supported lensFuncs for go.mod files.
func LensFuncs() map[string]source.LensFunc {
	return map[string]source.LensFunc{
		source.CommandUpgradeDependency.Name: upgradeLens,
		source.CommandTidy.Name:              tidyLens,
		source.CommandVendor.Name:            vendorLens,
	}
}

func upgradeLens(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) ([]protocol.CodeLens, error) {
	pm, err := snapshot.ParseMod(ctx, fh)
	if err != nil {
		return nil, err
	}
	module := pm.File.Module
	if module == nil || module.Syntax == nil {
		return nil, nil
	}
	upgrades, err := snapshot.ModUpgrade(ctx, fh)
	if err != nil {
		return nil, err
	}
	var (
		codelenses  []protocol.CodeLens
		allUpgrades []string
	)
	for _, req := range pm.File.Require {
		dep := req.Mod.Path
		latest, ok := upgrades[dep]
		if !ok {
			continue
		}
		if req.Syntax == nil {
			continue
		}
		// Get the range of the require directive.
		rng, err := positionsToRange(fh.URI(), pm.Mapper, req.Syntax.Start, req.Syntax.End)
		if err != nil {
			return nil, err
		}
		upgradeDepArgs, err := source.MarshalArgs(fh.URI(), []string{dep})
		if err != nil {
			return nil, err
		}
		codelenses = append(codelenses, protocol.CodeLens{
			Range: rng,
			Command: protocol.Command{
				Title:     fmt.Sprintf("Upgrade dependency to %s", latest),
				Command:   source.CommandUpgradeDependency.ID(),
				Arguments: upgradeDepArgs,
			},
		})
		allUpgrades = append(allUpgrades, dep)
	}
	// If there is at least 1 upgrade, add "Upgrade all dependencies" to
	// the module statement.
	if len(allUpgrades) > 0 {
		upgradeDepArgs, err := source.MarshalArgs(fh.URI(), append([]string{"-u"}, allUpgrades...))
		if err != nil {
			return nil, err
		}
		// Get the range of the module directive.
		moduleRng, err := positionsToRange(pm.Mapper.URI, pm.Mapper, module.Syntax.Start, module.Syntax.End)
		if err != nil {
			return nil, err
		}
		codelenses = append(codelenses, protocol.CodeLens{
			Range: moduleRng,
			Command: protocol.Command{
				Title:     "Upgrade all dependencies",
				Command:   source.CommandUpgradeDependency.ID(),
				Arguments: upgradeDepArgs,
			},
		})
	}
	return codelenses, err
}

func tidyLens(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) ([]protocol.CodeLens, error) {
	goModArgs, err := source.MarshalArgs(fh.URI())
	if err != nil {
		return nil, err
	}
	tidied, err := snapshot.ModTidy(ctx, fh)
	if err != nil {
		return nil, err
	}
	if len(tidied.Errors) == 0 {
		return nil, nil
	}
	pm, err := snapshot.ParseMod(ctx, fh)
	if err != nil {
		return nil, err
	}
	if pm.File == nil || pm.File.Module == nil || pm.File.Module.Syntax == nil {
		return nil, fmt.Errorf("no parsed go.mod for %s", fh.URI())
	}
	rng, err := positionsToRange(pm.Mapper.URI, pm.Mapper, pm.File.Module.Syntax.Start, pm.File.Module.Syntax.End)
	if err != nil {
		return nil, err
	}
	return []protocol.CodeLens{{
		Range: rng,
		Command: protocol.Command{
			Title:     "Tidy module",
			Command:   source.CommandTidy.ID(),
			Arguments: goModArgs,
		},
	}}, err
}

func vendorLens(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) ([]protocol.CodeLens, error) {
	goModArgs, err := source.MarshalArgs(fh.URI())
	if err != nil {
		return nil, err
	}
	pm, err := snapshot.ParseMod(ctx, fh)
	if err != nil {
		return nil, err
	}
	if pm.File == nil || pm.File.Module == nil || pm.File.Module.Syntax == nil {
		return nil, fmt.Errorf("no parsed go.mod for %s", fh.URI())
	}
	rng, err := positionsToRange(pm.Mapper.URI, pm.Mapper, pm.File.Module.Syntax.Start, pm.File.Module.Syntax.End)
	if err != nil {
		return nil, err
	}
	// Change the message depending on whether or not the module already has a
	// vendor directory.
	title := "Create vendor directory"
	vendorDir := filepath.Join(filepath.Dir(fh.URI().Filename()), "vendor")
	if info, _ := os.Stat(vendorDir); info != nil && info.IsDir() {
		title = "Sync vendor directory"
	}
	return []protocol.CodeLens{{
		Range: rng,
		Command: protocol.Command{
			Title:     title,
			Command:   source.CommandVendor.ID(),
			Arguments: goModArgs,
		},
	}}, nil
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
