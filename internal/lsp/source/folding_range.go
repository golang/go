package source

import (
	"context"
	"go/ast"
	"go/token"
	"sort"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
)

type FoldingRangeInfo struct {
	Range span.Range
	Kind  protocol.FoldingRangeKind
}

// FoldingRange gets all of the folding range for f.
func FoldingRange(ctx context.Context, view View, f GoFile, lineFoldingOnly bool) (ranges []FoldingRangeInfo, err error) {
	// TODO(suzmue): consider limiting the number of folding ranges returned, and
	// implement a way to prioritize folding ranges in that case.
	file, err := f.GetAST(ctx, ParseFull)
	if err != nil {
		return nil, err
	}

	// Get folding ranges for comments separately as they are not walked by ast.Inspect.
	ranges = append(ranges, commentsFoldingRange(f.FileSet(), file)...)

	visit := func(n ast.Node) bool {
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
			// Fold from position of "(" to position of ")".
			start, end = n.Lparen+1, n.Rparen
		}

		if start.IsValid() && end.IsValid() {
			if lineFoldingOnly && f.FileSet().Position(start).Line == f.FileSet().Position(end).Line {
				return true
			}
			ranges = append(ranges, FoldingRangeInfo{
				Range: span.NewRange(f.FileSet(), start, end),
				Kind:  kind,
			})
		}
		return true
	}

	// Walk the ast and collect folding ranges.
	ast.Inspect(file, visit)

	sort.Slice(ranges, func(i, j int) bool {
		if ranges[i].Range.Start < ranges[j].Range.Start {
			return true
		} else if ranges[i].Range.Start > ranges[j].Range.Start {
			return false
		}
		return ranges[i].Range.End < ranges[j].Range.End
	})
	return ranges, nil
}

// commentsFoldingRange returns the folding ranges for all comment blocks in file.
// The folding range starts at the end of the first comment, and ends at the end of the
// comment block and has kind protocol.Comment.
func commentsFoldingRange(fset *token.FileSet, file *ast.File) []FoldingRangeInfo {
	var comments []FoldingRangeInfo
	for _, commentGrp := range file.Comments {
		// Don't fold single comments.
		if len(commentGrp.List) <= 1 {
			continue
		}
		comments = append(comments, FoldingRangeInfo{
			// Fold from the end of the first line comment to the end of the comment block.
			Range: span.NewRange(fset, commentGrp.List[0].End(), commentGrp.End()),
			Kind:  protocol.Comment,
		})
	}
	return comments
}

func ToProtocolFoldingRanges(m *protocol.ColumnMapper, ranges []FoldingRangeInfo) ([]protocol.FoldingRange, error) {
	var res []protocol.FoldingRange
	for _, r := range ranges {
		spn, err := r.Range.Span()
		if err != nil {
			return nil, err
		}
		rng, err := m.Range(spn)
		if err != nil {
			return nil, err
		}
		res = append(res, protocol.FoldingRange{
			StartLine:      rng.Start.Line,
			StartCharacter: rng.Start.Character,
			EndLine:        rng.End.Line,
			EndCharacter:   rng.End.Character,
			Kind:           string(r.Kind),
		})
	}
	return res, nil
}
