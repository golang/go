// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package completion

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"go/types"
	"path/filepath"
	"strings"

	"golang.org/x/tools/internal/lsp/fuzzy"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

// packageClauseCompletions offers completions for a package declaration when
// one is not present in the given file.
func packageClauseCompletions(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle, pos protocol.Position) ([]CompletionItem, *Selection, error) {
	// We know that the AST for this file will be empty due to the missing
	// package declaration, but parse it anyway to get a mapper.
	pgf, err := snapshot.ParseGo(ctx, fh, source.ParseFull)
	if err != nil {
		return nil, nil, err
	}

	cursorSpan, err := pgf.Mapper.PointSpan(pos)
	if err != nil {
		return nil, nil, err
	}
	rng, err := cursorSpan.Range(pgf.Mapper.Converter)
	if err != nil {
		return nil, nil, err
	}

	surrounding, err := packageCompletionSurrounding(snapshot.FileSet(), fh, pgf, rng.Start)
	if err != nil {
		return nil, nil, errors.Errorf("invalid position for package completion: %w", err)
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

// packageCompletionSurrounding returns surrounding for package completion if a
// package completions can be suggested at a given position. A valid location
// for package completion is above any declarations or import statements.
func packageCompletionSurrounding(fset *token.FileSet, fh source.FileHandle, pgf *source.ParsedGoFile, pos token.Pos) (*Selection, error) {
	src, err := fh.Read()
	if err != nil {
		return nil, err
	}
	// If the file lacks a package declaration, the parser will return an empty
	// AST. As a work-around, try to parse an expression from the file contents.
	expr, _ := parser.ParseExprFrom(fset, fh.URI().Filename(), src, parser.Mode(0))
	if expr == nil {
		return nil, fmt.Errorf("unparseable file (%s)", fh.URI())
	}
	tok := fset.File(expr.Pos())
	cursor := tok.Pos(pgf.Tok.Offset(pos))
	m := &protocol.ColumnMapper{
		URI:       pgf.URI,
		Content:   src,
		Converter: span.NewContentConverter(fh.URI().Filename(), src),
	}

	// If we were able to parse out an identifier as the first expression from
	// the file, it may be the beginning of a package declaration ("pack ").
	// We can offer package completions if the cursor is in the identifier.
	if name, ok := expr.(*ast.Ident); ok {
		if cursor >= name.Pos() && cursor <= name.End() {
			if !strings.HasPrefix(PACKAGE, name.Name) {
				return nil, fmt.Errorf("cursor in non-matching ident")
			}
			return &Selection{
				content:     name.Name,
				cursor:      cursor,
				MappedRange: source.NewMappedRange(fset, m, name.Pos(), name.End()),
			}, nil
		}
	}

	// The file is invalid, but it contains an expression that we were able to
	// parse. We will use this expression to construct the cursor's
	// "surrounding".

	// First, consider the possibility that we have a valid "package" keyword
	// with an empty package name ("package "). "package" is parsed as an
	// *ast.BadDecl since it is a keyword. This logic would allow "package" to
	// appear on any line of the file as long as it's the first code expression
	// in the file.
	lines := strings.Split(string(src), "\n")
	cursorLine := tok.Line(cursor)
	if cursorLine <= 0 || cursorLine > len(lines) {
		return nil, fmt.Errorf("invalid line number")
	}
	if fset.Position(expr.Pos()).Line == cursorLine {
		words := strings.Fields(lines[cursorLine-1])
		if len(words) > 0 && words[0] == PACKAGE {
			content := PACKAGE
			// Account for spaces if there are any.
			if len(words) > 1 {
				content += " "
			}

			start := expr.Pos()
			end := token.Pos(int(expr.Pos()) + len(content) + 1)
			// We have verified that we have a valid 'package' keyword as our
			// first expression. Ensure that cursor is in this keyword or
			// otherwise fallback to the general case.
			if cursor >= start && cursor <= end {
				return &Selection{
					content:     content,
					cursor:      cursor,
					MappedRange: source.NewMappedRange(fset, m, start, end),
				}, nil
			}
		}
	}

	// If the cursor is after the start of the expression, no package
	// declaration will be valid.
	if cursor > expr.Pos() {
		return nil, fmt.Errorf("cursor after expression")
	}

	// If the cursor is in a comment, don't offer any completions.
	if cursorInComment(fset, cursor, src) {
		return nil, fmt.Errorf("cursor in comment")
	}

	// The surrounding range in this case is the cursor except for empty file,
	// in which case it's end of file - 1
	start, end := cursor, cursor
	if tok.Size() == 0 {
		start, end = tok.Pos(0)-1, tok.Pos(0)-1
	}

	return &Selection{
		content:     "",
		cursor:      cursor,
		MappedRange: source.NewMappedRange(fset, m, start, end),
	}, nil
}

func cursorInComment(fset *token.FileSet, cursor token.Pos, src []byte) bool {
	var s scanner.Scanner
	s.Init(fset.File(cursor), src, func(_ token.Position, _ string) {}, scanner.ScanComments)
	for {
		pos, tok, lit := s.Scan()
		if pos <= cursor && cursor <= token.Pos(int(pos)+len(lit)) {
			return tok == token.COMMENT
		}
		if tok == token.EOF {
			break
		}
	}
	return false
}

// packageNameCompletions returns name completions for a package clause using
// the current name as prefix.
func (c *completer) packageNameCompletions(ctx context.Context, fileURI span.URI, name *ast.Ident) error {
	cursor := int(c.pos - name.NamePos)
	if cursor < 0 || cursor > len(name.Name) {
		return errors.New("cursor is not in package name identifier")
	}

	c.completionContext.packageCompletion = true

	prefix := name.Name[:cursor]
	packageSuggestions, err := packageSuggestions(ctx, c.snapshot, fileURI, prefix)
	if err != nil {
		return err
	}

	for _, pkg := range packageSuggestions {
		c.deepState.enqueue(pkg)
	}
	return nil
}

// packageSuggestions returns a list of packages from workspace packages that
// have the given prefix and are used in the same directory as the given
// file. This also includes test packages for these packages (<pkg>_test) and
// the directory name itself.
func packageSuggestions(ctx context.Context, snapshot source.Snapshot, fileURI span.URI, prefix string) ([]candidate, error) {
	workspacePackages, err := snapshot.WorkspacePackages(ctx)
	if err != nil {
		return nil, err
	}

	dirPath := filepath.Dir(string(fileURI))
	dirName := filepath.Base(dirPath)

	seenPkgs := make(map[string]struct{})

	toCandidate := func(name string, score float64) candidate {
		obj := types.NewPkgName(0, nil, name, types.NewPackage("", name))
		return candidate{obj: obj, name: name, detail: name, score: score}
	}

	matcher := fuzzy.NewMatcher(prefix)

	// The `go` command by default only allows one package per directory but we
	// support multiple package suggestions since gopls is build system agnostic.
	var packages []candidate
	for _, pkg := range workspacePackages {
		if pkg.Name() == "main" || pkg.Name() == "" {
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
