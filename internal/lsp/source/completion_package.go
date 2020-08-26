// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/types"
	"path/filepath"
	"strings"

	"golang.org/x/tools/internal/lsp/fuzzy"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

// packageClauseCompletions offers completions for a package declaration when
// one is not present in the given file.
func packageClauseCompletions(ctx context.Context, snapshot Snapshot, fh FileHandle, pos protocol.Position) ([]CompletionItem, *Selection, error) {
	// We know that the AST for this file will be empty due to the missing
	// package declaration, but parse it anyway to get a mapper.
	pgf, err := snapshot.ParseGo(ctx, fh, ParseHeader)
	if err != nil {
		return nil, nil, err
	}

	// Check that the file is completely empty, to avoid offering incorrect package
	// clause completions.
	// TODO: Support package clause completions in all files.
	if pgf.Tok.Size() != 0 {
		return nil, nil, errors.New("package clause completion is only offered for empty file")
	}

	cursorSpan, err := pgf.Mapper.PointSpan(pos)
	if err != nil {
		return nil, nil, err
	}
	rng, err := cursorSpan.Range(pgf.Mapper.Converter)
	if err != nil {
		return nil, nil, err
	}

	surrounding := &Selection{
		content:     "",
		cursor:      rng.Start,
		mappedRange: newMappedRange(snapshot.FileSet(), pgf.Mapper, rng.Start, rng.Start),
	}

	packageSuggestions, err := packageSuggestions(ctx, snapshot, fh.URI(), "")
	if err != nil {
		return nil, nil, err
	}

	var items []CompletionItem
	for _, pkg := range packageSuggestions {
		insertText := fmt.Sprintf("package %s", pkg.name)
		items = append(items, CompletionItem{
			Label:      insertText,
			Kind:       protocol.ModuleCompletion,
			InsertText: insertText,
			Score:      pkg.score,
		})
	}

	return items, surrounding, nil
}

// packageNameCompletions returns name completions for a package clause using
// the current name as prefix.
func (c *completer) packageNameCompletions(ctx context.Context, fileURI span.URI, name *ast.Ident) error {
	cursor := int(c.pos - name.NamePos)
	if cursor < 0 || cursor > len(name.Name) {
		return errors.New("cursor is not in package name identifier")
	}

	prefix := name.Name[:cursor]
	packageSuggestions, err := packageSuggestions(ctx, c.snapshot, fileURI, prefix)
	if err != nil {
		return err
	}

	for _, pkg := range packageSuggestions {
		if item, err := c.item(ctx, pkg); err == nil {
			c.items = append(c.items, item)
		}
	}
	return nil
}

// packageSuggestions returns a list of packages from workspace packages that
// have the given prefix and are used in the the same directory as the given
// file. This also includes test packages for these packages (<pkg>_test) and
// the directory name itself.
func packageSuggestions(ctx context.Context, snapshot Snapshot, fileURI span.URI, prefix string) ([]candidate, error) {
	workspacePackages, err := snapshot.WorkspacePackages(ctx)
	if err != nil {
		return nil, err
	}

	dirPath := filepath.Dir(string(fileURI))
	dirName := filepath.Base(dirPath)

	seenPkgs := make(map[string]struct{})

	toCandidate := func(name string, score float64) candidate {
		obj := types.NewPkgName(0, nil, name, types.NewPackage("", name))
		return candidate{obj: obj, name: name, score: score}
	}

	matcher := fuzzy.NewMatcher(prefix)

	// The `go` command by default only allows one package per directory but we
	// support multiple package suggestions since gopls is build system agnostic.
	var packages []candidate
	for _, pkg := range workspacePackages {
		if pkg.Name() == "main" {
			continue
		}
		if _, ok := seenPkgs[pkg.Name()]; ok {
			continue
		}

		// Only add packages that are previously used in the current directory.
		var relevantPkg bool
		for _, pgf := range pkg.CompiledGoFiles() {
			if filepath.Dir(string(pgf.URI)) == dirPath {
				relevantPkg = true
				break
			}
		}
		if !relevantPkg {
			continue
		}

		// Add a found package used in current directory as a high relevance
		// suggestion and the test package for it as a medium relevance
		// suggestion.
		if score := float64(matcher.Score(pkg.Name())); score > 0 {
			packages = append(packages, toCandidate(pkg.Name(), score*highScore))
		}
		seenPkgs[pkg.Name()] = struct{}{}

		testPkgName := pkg.Name() + "_test"
		if _, ok := seenPkgs[testPkgName]; ok || strings.HasSuffix(pkg.Name(), "_test") {
			continue
		}
		if score := float64(matcher.Score(testPkgName)); score > 0 {
			packages = append(packages, toCandidate(testPkgName, score*stdScore))
		}
		seenPkgs[testPkgName] = struct{}{}
	}

	// Add current directory name as a low relevance suggestion.
	if _, ok := seenPkgs[dirName]; !ok {
		if score := float64(matcher.Score(dirName)); score > 0 {
			packages = append(packages, toCandidate(dirName, score*lowScore))
		}

		testDirName := dirName + "_test"
		if score := float64(matcher.Score(testDirName)); score > 0 {
			packages = append(packages, toCandidate(testDirName, score*lowScore))
		}
	}

	if score := float64(matcher.Score("main")); score > 0 {
		packages = append(packages, toCandidate("main", score*lowScore))
	}

	return packages, nil
}
