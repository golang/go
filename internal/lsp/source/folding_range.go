package source

import (
	"context"
	"go/ast"
	"go/token"
	"sort"

	"golang.org/x/tools/internal/lsp/protocol"
)

type FoldingRangeInfo struct {
	mappedRange
	Kind protocol.FoldingRangeKind
}

// FoldingRange gets all of the folding range for f.
func FoldingRange(ctx context.Context, snapshot Snapshot, fh FileHandle, lineFoldingOnly bool) (ranges []*FoldingRangeInfo, err error) {
	// TODO(suzmue): consider limiting the number of folding ranges returned, and
	// implement a way to prioritize folding ranges in that case.
	pgh := snapshot.View().Session().Cache().ParseGoHandle(fh, ParseFull)
	file, m, _, err := pgh.Parse(ctx)
	if err != nil {
		return nil, err
	}
	fset := snapshot.View().Session().Cache().FileSet()

	// Get folding ranges for comments separately as they are not walked by ast.Inspect.
	ranges = append(ranges, commentsFoldingRange(fset, m, file)...)

	foldingFunc := foldingRange
	if lineFoldingOnly {
		foldingFunc = lineFoldingRange
	}

	visit := func(n ast.Node) bool {
		rng := foldingFunc(fset, m, n)
		if rng != nil {
			ranges = append(ranges, rng)
		}
		return true
	}
	// Walk the ast and collect folding ranges.
	ast.Inspect(file, visit)

	sort.Slice(ranges, func(i, j int) bool {
		irng, _ := ranges[i].Range()
		jrng, _ := ranges[j].Range()
		return protocol.CompareRange(irng, jrng) < 0
	})

	return ranges, nil
}

// foldingRange calculates the folding range for n.
func foldingRange(fset *token.FileSet, m *protocol.ColumnMapper, n ast.Node) *FoldingRangeInfo {
	var kind protocol.FoldingRangeKind
	var start, end token.Pos
	switch n := n.(type) {
	case *ast.BlockStmt:
		// Fold from position of "{" to position of "}".
		start, end = n.Lbrace+1, n.Rbrace
	case *ast.CaseClause:
		// Fold from position of ":" to end.
		start, end = n.Colon+1, n.End()
	case *ast.CallExpr:
		// Fold from position of "(" to position of ")".
		start, end = n.Lparen+1, n.Rparen
	case *ast.FieldList:
		// Fold from position of opening parenthesis/brace, to position of
		// closing parenthesis/brace.
		start, end = n.Opening+1, n.Closing
	case *ast.GenDecl:
		// If this is an import declaration, set the kind to be protocol.Imports.
		if n.Tok == token.IMPORT {
			kind = protocol.Imports
		}
		start, end = n.Lparen+1, n.Rparen
	}
	if !start.IsValid() || !end.IsValid() {
		return nil
	}
	return &FoldingRangeInfo{
		mappedRange: newMappedRange(fset, m, start, end),
		Kind:        kind,
	}
}

// lineFoldingRange calculates the line folding range for n.
func lineFoldingRange(fset *token.FileSet, m *protocol.ColumnMapper, n ast.Node) *FoldingRangeInfo {

	// TODO(suzmue): include trailing empty lines before the closing
	// parenthesis/brace.
	var kind protocol.FoldingRangeKind
	var start, end token.Pos
	switch n := n.(type) {
	case *ast.BlockStmt:
		// Fold lines between "{" and "}".
		if !n.Lbrace.IsValid() || !n.Rbrace.IsValid() {
			break
		}
		nStmts := len(n.List)
		if nStmts == 0 {
			break
		}
		// Don't want to fold if the start is on the same line as the brace.
		if fset.Position(n.Lbrace).Line == fset.Position(n.List[0].Pos()).Line {
			break
		}
		// Don't want to fold if the end is on the same line as the brace.
		if fset.Position(n.Rbrace).Line == fset.Position(n.List[nStmts-1].End()).Line {
			break
		}
		start, end = n.Lbrace+1, n.List[nStmts-1].End()
	case *ast.CaseClause:
		// Fold from position of ":" to end.
		start, end = n.Colon+1, n.End()
	case *ast.FieldList:
		// Fold lines between opening parenthesis/brace and closing parenthesis/brace.
		if !n.Opening.IsValid() || !n.Closing.IsValid() {
			break
		}
		nFields := len(n.List)
		if nFields == 0 {
			break
		}
		// Don't want to fold if the start is on the same line as the parenthesis/brace.
		if fset.Position(n.Opening).Line == fset.Position(n.List[nFields-1].End()).Line {
			break
		}
		// Don't want to fold if the end is on the same line as the parenthesis/brace.
		if fset.Position(n.Closing).Line == fset.Position(n.List[nFields-1].End()).Line {
			break
		}
		start, end = n.Opening+1, n.List[nFields-1].End()
	case *ast.GenDecl:
		// If this is an import declaration, set the kind to be protocol.Imports.
		if n.Tok == token.IMPORT {
			kind = protocol.Imports
		}
		// Fold from position of "(" to position of ")".
		if !n.Lparen.IsValid() || !n.Rparen.IsValid() {
			break
		}
		nSpecs := len(n.Specs)
		if nSpecs == 0 {
			break
		}
		// Don't want to fold if the end is on the same line as the parenthesis/brace.
		if fset.Position(n.Lparen).Line == fset.Position(n.Specs[0].Pos()).Line {
			break
		}
		// Don't want to fold if the end is on the same line as the parenthesis/brace.
		if fset.Position(n.Rparen).Line == fset.Position(n.Specs[nSpecs-1].End()).Line {
			break
		}
		start, end = n.Lparen+1, n.Specs[nSpecs-1].End()
	}

	// Check that folding positions are valid.
	if !start.IsValid() || !end.IsValid() {
		return nil
	}
	// Do not fold if the start and end lines are the same.
	if fset.Position(start).Line == fset.Position(end).Line {
		return nil
	}
	return &FoldingRangeInfo{
		mappedRange: newMappedRange(fset, m, start, end),
		Kind:        kind,
	}
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
			mappedRange: newMappedRange(fset, m, commentGrp.List[0].End(), commentGrp.End()),
			Kind:        protocol.Comment,
		})
	}
	return comments
}
