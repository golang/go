// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/ast"
	"go/token"
	"sort"

	"golang.org/x/tools/internal/lsp/protocol"
)

// FoldingRangeInfo holds range and kind info of folding for an ast.Node
type FoldingRangeInfo struct {
	MappedRange
	Kind protocol.FoldingRangeKind
}

// FoldingRange gets all of the folding range for f.
func FoldingRange(ctx context.Context, snapshot Snapshot, fh FileHandle, lineFoldingOnly bool) (ranges []*FoldingRangeInfo, err error) {
	// TODO(suzmue): consider limiting the number of folding ranges returned, and
	// implement a way to prioritize folding ranges in that case.
	pgf, err := snapshot.ParseGo(ctx, fh, ParseFull)
	if err != nil {
		return nil, err
	}
	fset := snapshot.FileSet()

	// Get folding ranges for comments separately as they are not walked by ast.Inspect.
	ranges = append(ranges, commentsFoldingRange(fset, pgf.Mapper, pgf.File)...)

	visit := func(n ast.Node) bool {
		rng := foldingRangeFunc(fset, pgf.Mapper, n, lineFoldingOnly)
		if rng != nil {
			ranges = append(ranges, rng)
		}
		return true
	}
	// Walk the ast and collect folding ranges.
	ast.Inspect(pgf.File, visit)

	sort.Slice(ranges, func(i, j int) bool {
		irng, _ := ranges[i].Range()
		jrng, _ := ranges[j].Range()
		return protocol.CompareRange(irng, jrng) < 0
	})

	return ranges, nil
}

// foldingRangeFunc calculates the line folding range for ast.Node n
func foldingRangeFunc(fset *token.FileSet, m *protocol.ColumnMapper, n ast.Node, lineFoldingOnly bool) *FoldingRangeInfo {
	// TODO(suzmue): include trailing empty lines before the closing
	// parenthesis/brace.
	var kind protocol.FoldingRangeKind
	var start, end token.Pos
	switch n := n.(type) {
	case *ast.BlockStmt:
		// Fold between positions of or lines between "{" and "}".
		var startList, endList token.Pos
		if num := len(n.List); num != 0 {
			startList, endList = n.List[0].Pos(), n.List[num-1].End()
		}
		start, end = validLineFoldingRange(fset, n.Lbrace, n.Rbrace, startList, endList, lineFoldingOnly)
	case *ast.CaseClause:
		// Fold from position of ":" to end.
		start, end = n.Colon+1, n.End()
	case *ast.CommClause:
		// Fold from position of ":" to end.
		start, end = n.Colon+1, n.End()
	case *ast.CallExpr:
		// Fold from position of "(" to position of ")".
		start, end = n.Lparen+1, n.Rparen
	case *ast.FieldList:
		// Fold between positions of or lines between opening parenthesis/brace and closing parenthesis/brace.
		var startList, endList token.Pos
		if num := len(n.List); num != 0 {
			startList, endList = n.List[0].Pos(), n.List[num-1].End()
		}
		start, end = validLineFoldingRange(fset, n.Opening, n.Closing, startList, endList, lineFoldingOnly)
	case *ast.GenDecl:
		// If this is an import declaration, set the kind to be protocol.Imports.
		if n.Tok == token.IMPORT {
			kind = protocol.Imports
		}
		// Fold between positions of or lines between "(" and ")".
		var startSpecs, endSpecs token.Pos
		if num := len(n.Specs); num != 0 {
			startSpecs, endSpecs = n.Specs[0].Pos(), n.Specs[num-1].End()
		}
		start, end = validLineFoldingRange(fset, n.Lparen, n.Rparen, startSpecs, endSpecs, lineFoldingOnly)
	case *ast.CompositeLit:
		// Fold between positions of or lines between "{" and "}".
		var startElts, endElts token.Pos
		if num := len(n.Elts); num != 0 {
			startElts, endElts = n.Elts[0].Pos(), n.Elts[num-1].End()
		}
		start, end = validLineFoldingRange(fset, n.Lbrace, n.Rbrace, startElts, endElts, lineFoldingOnly)
	}

	// Check that folding positions are valid.
	if !start.IsValid() || !end.IsValid() {
		return nil
	}
	// in line folding mode, do not fold if the start and end lines are the same.
	if lineFoldingOnly && fset.Position(start).Line == fset.Position(end).Line {
		return nil
	}
	return &FoldingRangeInfo{
		MappedRange: NewMappedRange(fset, m, start, end),
		Kind:        kind,
	}
}

// validLineFoldingRange returns start and end token.Pos for folding range if the range is valid.
// returns token.NoPos otherwise, which fails token.IsValid check
func validLineFoldingRange(fset *token.FileSet, open, close, start, end token.Pos, lineFoldingOnly bool) (token.Pos, token.Pos) {
	if lineFoldingOnly {
		if !open.IsValid() || !close.IsValid() {
			return token.NoPos, token.NoPos
		}

		// Don't want to fold if the start/end is on the same line as the open/close
		// as an example, the example below should *not* fold:
		// var x = [2]string{"d",
		// "e" }
		if fset.Position(open).Line == fset.Position(start).Line ||
			fset.Position(close).Line == fset.Position(end).Line {
			return token.NoPos, token.NoPos
		}

		return open + 1, end
	}
	return open + 1, close
}

// commentsFoldingRange returns the folding ranges for all comment blocks in file.
// The folding range starts at the end of the first comment, and ends at the end of the
// comment block and has kind protocol.Comment.
func commentsFoldingRange(fset *token.FileSet, m *protocol.ColumnMapper, file *ast.File) (comments []*FoldingRangeInfo) {
	for _, commentGrp := range file.Comments {
		// Don't fold single comments.
		if len(commentGrp.List) <= 1 {
			continue
		}
		comments = append(comments, &FoldingRangeInfo{
			// Fold from the end of the first line comment to the end of the comment block.
			MappedRange: NewMappedRange(fset, m, commentGrp.List[0].End(), commentGrp.End()),
			Kind:        protocol.Comment,
		})
	}
	return comments
}
