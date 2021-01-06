// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
		source.CommandUpgradeDependency.Name: upgradeLenses,
		source.CommandTidy.Name:              tidyLens,
		source.CommandVendor.Name:            vendorLens,
	}
}

func upgradeLenses(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) ([]protocol.CodeLens, error) {
	pm, err := snapshot.ParseMod(ctx, fh)
	if err != nil || pm.File == nil {
		return nil, err
	}
	if len(pm.File.Require) == 0 {
		// Nothing to upgrade.
		return nil, nil
	}
	upgradeTransitiveArgs, err := source.MarshalArgs(fh.URI(), false, []string{"-u", "all"})
	if err != nil {
		return nil, err
	}
	var requires []string
	for _, req := range pm.File.Require {
		requires = append(requires, req.Mod.Path)
	}
	upgradeDirectArgs, err := source.MarshalArgs(fh.URI(), false, requires)
	if err != nil {
		return nil, err
	}
	// Put the upgrade code lenses above the first require block or statement.
	rng, err := firstRequireRange(fh, pm)
	if err != nil {
		return nil, err
	}
	return []protocol.CodeLens{
		{
			Range: rng,
			Command: protocol.Command{
				Title:     "Upgrade transitive dependencies",
				Command:   source.CommandUpgradeDependency.ID(),
				Arguments: upgradeTransitiveArgs,
			},
		},
		{
			Range: rng,
			Command: protocol.Command{
				Title:     "Upgrade direct dependencies",
				Command:   source.CommandUpgradeDependency.ID(),
				Arguments: upgradeDirectArgs,
			},
		},
	}, nil

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
	return lineToRange(pm.Mapper, fh.URI(), syntax.Start, syntax.End)
}

// firstRequireRange returns the range for the first "require" in the given
// go.mod file. This is either a require block or an individual require line.
func firstRequireRange(fh source.FileHandle, pm *source.ParsedModule) (protocol.Range, error) {
	if len(pm.File.Require) == 0 {
		return protocol.Range{}, fmt.Errorf("no requires in the file %s", fh.URI())
	}
	var start, end modfile.Position
	for _, stmt := range pm.File.Syntax.Stmt {
		if b, ok := stmt.(*modfile.LineBlock); ok && len(b.Token) == 1 && b.Token[0] == "require" {
			start, end = b.Span()
			break
		}
	}

	firstRequire := pm.File.Require[0].Syntax
	if start.Byte == 0 || firstRequire.Start.Byte < start.Byte {
		start, end = firstRequire.Start, firstRequire.End
	}
	return lineToRange(pm.Mapper, fh.URI(), start, end)
}

func lineToRange(m *protocol.ColumnMapper, uri span.URI, start, end modfile.Position) (protocol.Range, error) {
	line, col, err := m.Converter.ToPosition(start.Byte)
	if err != nil {
		return protocol.Range{}, err
	}
	s := span.NewPoint(line, col, start.Byte)
	line, col, err = m.Converter.ToPosition(end.Byte)
	if err != nil {
		return protocol.Range{}, err
	}
	e := span.NewPoint(line, col, end.Byte)
	return m.Range(span.New(uri, s, e))
}
