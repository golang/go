// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/ast"
	"go/token"
	"sort"
	"strings"

	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
)

// FoldingRangeInfo holds range and kind info of folding for an ast.Node
type FoldingRangeInfo struct {
	MappedRange protocol.MappedRange
	Kind        protocol.FoldingRangeKind
}

// FoldingRange gets all of the folding range for f.
func FoldingRange(ctx context.Context, snapshot Snapshot, fh FileHandle, lineFoldingOnly bool) (ranges []*FoldingRangeInfo, err error) {
	// TODO(suzmue): consider limiting the number of folding ranges returned, and
	// implement a way to prioritize folding ranges in that case.
	pgf, err := snapshot.ParseGo(ctx, fh, ParseFull)
	if err != nil {
		return nil, err
	}

	// With parse errors, we wouldn't be able to produce accurate folding info.
	// LSP protocol (3.16) currently does not have a way to handle this case
	// (https://github.com/microsoft/language-server-protocol/issues/1200).
	// We cannot return an error either because we are afraid some editors
	// may not handle errors nicely. As a workaround, we now return an empty
	// result and let the client handle this case by double check the file
	// contents (i.e. if the file is not empty and the folding range result
	// is empty, raise an internal error).
	if pgf.ParseErr != nil {
		return nil, nil
	}

	// Get folding ranges for comments separately as they are not walked by ast.Inspect.
	ranges = append(ranges, commentsFoldingRange(pgf)...)

	visit := func(n ast.Node) bool {
		rng := foldingRangeFunc(pgf, n, lineFoldingOnly)
		if rng != nil {
			ranges = append(ranges, rng)
		}
		return true
	}
	// Walk the ast and collect folding ranges.
	ast.Inspect(pgf.File, visit)

	sort.Slice(ranges, func(i, j int) bool {
		irng := ranges[i].MappedRange.Range()
		jrng := ranges[j].MappedRange.Range()
		return protocol.CompareRange(irng, jrng) < 0
	})

	return ranges, nil
}

// foldingRangeFunc calculates the line folding range for ast.Node n
func foldingRangeFunc(pgf *ParsedGoFile, n ast.Node, lineFoldingOnly bool) *FoldingRangeInfo {
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
		start, end = validLineFoldingRange(pgf.Tok, n.Lbrace, n.Rbrace, startList, endList, lineFoldingOnly)
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
		start, end = validLineFoldingRange(pgf.Tok, n.Opening, n.Closing, startList, endList, lineFoldingOnly)
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
		start, end = validLineFoldingRange(pgf.Tok, n.Lparen, n.Rparen, startSpecs, endSpecs, lineFoldingOnly)
	case *ast.BasicLit:
		// Fold raw string literals from position of "`" to position of "`".
		if n.Kind == token.STRING && len(n.Value) >= 2 && n.Value[0] == '`' && n.Value[len(n.Value)-1] == '`' {
			start, end = n.Pos(), n.End()
		}
	case *ast.CompositeLit:
		// Fold between positions of or lines between "{" and "}".
		var startElts, endElts token.Pos
		if num := len(n.Elts); num != 0 {
			startElts, endElts = n.Elts[0].Pos(), n.Elts[num-1].End()
		}
		start, end = validLineFoldingRange(pgf.Tok, n.Lbrace, n.Rbrace, startElts, endElts, lineFoldingOnly)
	}

	// Check that folding positions are valid.
	if !start.IsValid() || !end.IsValid() {
		return nil
	}
	// in line folding mode, do not fold if the start and end lines are the same.
	if lineFoldingOnly && safetoken.Line(pgf.Tok, start) == safetoken.Line(pgf.Tok, end) {
		return nil
	}
	mrng, err := pgf.PosMappedRange(start, end)
	if err != nil {
		bug.Errorf("%w", err) // can't happen
	}
	return &FoldingRangeInfo{
		MappedRange: mrng,
		Kind:        kind,
	}
}

// validLineFoldingRange returns start and end token.Pos for folding range if the range is valid.
// returns token.NoPos otherwise, which fails token.IsValid check
func validLineFoldingRange(tokFile *token.File, open, close, start, end token.Pos, lineFoldingOnly bool) (token.Pos, token.Pos) {
	if lineFoldingOnly {
		if !open.IsValid() || !close.IsValid() {
			return token.NoPos, token.NoPos
		}

		// Don't want to fold if the start/end is on the same line as the open/close
		// as an example, the example below should *not* fold:
		// var x = [2]string{"d",
		// "e" }
		if safetoken.Line(tokFile, open) == safetoken.Line(tokFile, start) ||
			safetoken.Line(tokFile, close) == safetoken.Line(tokFile, end) {
			return token.NoPos, token.NoPos
		}

		return open + 1, end
	}
	return open + 1, close
}

// commentsFoldingRange returns the folding ranges for all comment blocks in file.
// The folding range starts at the end of the first line of the comment block, and ends at the end of the
// comment block and has kind protocol.Comment.
func commentsFoldingRange(pgf *ParsedGoFile) (comments []*FoldingRangeInfo) {
	tokFile := pgf.Tok
	for _, commentGrp := range pgf.File.Comments {
		startGrpLine, endGrpLine := safetoken.Line(tokFile, commentGrp.Pos()), safetoken.Line(tokFile, commentGrp.End())
		if startGrpLine == endGrpLine {
			// Don't fold single line comments.
			continue
		}

		firstComment := commentGrp.List[0]
		startPos, endLinePos := firstComment.Pos(), firstComment.End()
		startCmmntLine, endCmmntLine := safetoken.Line(tokFile, startPos), safetoken.Line(tokFile, endLinePos)
		if startCmmntLine != endCmmntLine {
			// If the first comment spans multiple lines, then we want to have the
			// folding range start at the end of the first line.
			endLinePos = token.Pos(int(startPos) + len(strings.Split(firstComment.Text, "\n")[0]))
		}
		mrng, err := pgf.PosMappedRange(endLinePos, commentGrp.End())
		if err != nil {
			bug.Errorf("%w", err) // can't happen
		}
		comments = append(comments, &FoldingRangeInfo{
			// Fold from the end of the first line comment to the end of the comment block.
			MappedRange: mrng,
			Kind:        protocol.Comment,
		})
	}
	return comments
}
