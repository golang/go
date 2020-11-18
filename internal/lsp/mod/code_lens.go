package mod

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

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
	if err != nil || pm.File == nil {
		return nil, err
	}
	if len(pm.File.Require) == 0 {
		// Nothing to upgrade.
		return nil, nil
	}
	upgradeDepArgs, err := source.MarshalArgs(fh.URI(), false, []string{"-u", "all"})
	if err != nil {
		return nil, err
	}
	rng, err := moduleStmtRange(fh, pm)
	if err != nil {
		return nil, err
	}
	return []protocol.CodeLens{{
		Range: rng,
		Command: protocol.Command{
			Title:     "Upgrade all dependencies",
			Command:   source.CommandUpgradeDependency.ID(),
			Arguments: upgradeDepArgs,
		},
	}}, nil

}

func tidyLens(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) ([]protocol.CodeLens, error) {
	pm, err := snapshot.ParseMod(ctx, fh)
	if err != nil || pm.File == nil {
		return nil, err
	}
	if len(pm.File.Require) == 0 {
		// Nothing to vendor.
		return nil, nil
	}
	goModArgs, err := source.MarshalArgs(fh.URI())
	if err != nil {
		return nil, err
	}
	rng, err := moduleStmtRange(fh, pm)
	if err != nil {
		return nil, err
	}
	return []protocol.CodeLens{{
		Range: rng,
		Command: protocol.Command{
			Title:     source.CommandTidy.Title,
			Command:   source.CommandTidy.ID(),
			Arguments: goModArgs,
		},
	}}, nil
}

func vendorLens(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) ([]protocol.CodeLens, error) {
	pm, err := snapshot.ParseMod(ctx, fh)
	if err != nil || pm.File == nil {
		return nil, err
	}
	rng, err := moduleStmtRange(fh, pm)
	if err != nil {
		return nil, err
	}
	goModArgs, err := source.MarshalArgs(fh.URI())
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

func moduleStmtRange(fh source.FileHandle, pm *source.ParsedModule) (protocol.Range, error) {
	if pm.File == nil || pm.File.Module == nil || pm.File.Module.Syntax == nil {
		return protocol.Range{}, fmt.Errorf("no module statement in %s", fh.URI())
	}
	syntax := pm.File.Module.Syntax
	line, col, err := pm.Mapper.Converter.ToPosition(syntax.Start.Byte)
	if err != nil {
		return protocol.Range{}, err
	}
	start := span.NewPoint(line, col, syntax.Start.Byte)
	line, col, err = pm.Mapper.Converter.ToPosition(syntax.End.Byte)
	if err != nil {
		return protocol.Range{}, err
	}
	end := span.NewPoint(line, col, syntax.End.Byte)
	rng, err := pm.Mapper.Range(span.New(fh.URI(), start, end))
	if err != nil {
		return protocol.Range{}, err
	}
	return rng, err
}
