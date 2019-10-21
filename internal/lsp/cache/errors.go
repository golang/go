package cache

import (
	"bytes"
	"context"
	"go/scanner"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func sourceError(ctx context.Context, pkg *pkg, e interface{}) (*source.Error, error) {
	var (
		spn           span.Span
		err           error
		msg, category string
		kind          source.ErrorKind
		fixes         []source.SuggestedFix
		related       []source.RelatedInformation
	)
	switch e := e.(type) {
	case packages.Error:
		if e.Pos == "" {
			spn = parseGoListError(e.Msg)
		} else {
			spn = span.Parse(e.Pos)
		}
		msg = e.Msg
		kind = toSourceErrorKind(e.Kind)
	case *scanner.Error:
		msg = e.Msg
		kind = source.ParseError
		spn = span.Parse(e.Pos.String())
	case scanner.ErrorList:
		// The first parser error is likely the root cause of the problem.
		if e.Len() > 0 {
			spn = span.Parse(e[0].Pos.String())
			msg = e[0].Msg
			kind = source.ParseError
		}
	case types.Error:
		spn = span.Parse(pkg.snapshot.view.session.cache.fset.Position(e.Pos).String())
		msg = e.Msg
		kind = source.TypeError
	case *analysis.Diagnostic:
		spn, err = span.NewRange(pkg.snapshot.view.session.cache.fset, e.Pos, e.End).Span()
		if err != nil {
			return nil, err
		}
		msg = e.Message
		kind = source.Analysis
		category = e.Category
		fixes, err = suggestedFixes(ctx, pkg, e)
		if err != nil {
			return nil, err
		}
		related, err = relatedInformation(ctx, pkg, e)
		if err != nil {
			return nil, err
		}
	}
	rng, err := spanToRange(ctx, pkg, spn, kind == source.TypeError)
	if err != nil {
		return nil, err
	}
	return &source.Error{
		URI:            spn.URI(),
		Range:          rng,
		Message:        msg,
		Kind:           kind,
		Category:       category,
		SuggestedFixes: fixes,
		Related:        related,
	}, nil
}

func suggestedFixes(ctx context.Context, pkg *pkg, diag *analysis.Diagnostic) ([]source.SuggestedFix, error) {
	var fixes []source.SuggestedFix
	for _, fix := range diag.SuggestedFixes {
		edits := make(map[span.URI][]protocol.TextEdit)
		for _, e := range fix.TextEdits {
			spn, err := span.NewRange(pkg.Snapshot().View().Session().Cache().FileSet(), e.Pos, e.End).Span()
			if err != nil {
				return nil, err
			}
			rng, err := spanToRange(ctx, pkg, spn, false)
			if err != nil {
				return nil, err
			}
			edits[spn.URI()] = append(edits[spn.URI()], protocol.TextEdit{
				Range:   rng,
				NewText: string(e.NewText),
			})
		}
		fixes = append(fixes, source.SuggestedFix{
			Title: fix.Message,
			Edits: edits,
		})
	}
	return fixes, nil
}

func relatedInformation(ctx context.Context, pkg *pkg, diag *analysis.Diagnostic) ([]source.RelatedInformation, error) {
	var out []source.RelatedInformation
	for _, related := range diag.Related {
		spn, err := span.NewRange(pkg.Snapshot().View().Session().Cache().FileSet(), related.Pos, related.End).Span()
		if err != nil {
			return nil, err
		}
		rng, err := spanToRange(ctx, pkg, spn, false)
		if err != nil {
			return nil, err
		}
		out = append(out, source.RelatedInformation{
			URI:     spn.URI(),
			Range:   rng,
			Message: related.Message,
		})
	}
	return out, nil
}

func toSourceErrorKind(kind packages.ErrorKind) source.ErrorKind {
	switch kind {
	case packages.ListError:
		return source.ListError
	case packages.ParseError:
		return source.ParseError
	case packages.TypeError:
		return source.TypeError
	default:
		return source.UnknownError
	}
}

// spanToRange converts a span.Span to a protocol.Range,
// assuming that the span belongs to the package whose diagnostics are being computed.
func spanToRange(ctx context.Context, pkg *pkg, spn span.Span, isTypeError bool) (protocol.Range, error) {
	ph, err := pkg.File(spn.URI())
	if err != nil {
		return protocol.Range{}, err
	}
	_, m, _, err := ph.Cached(ctx)
	if err != nil {
		return protocol.Range{}, err
	}
	if spn.IsPoint() && isTypeError {
		data, _, err := ph.File().Read(ctx)
		if err != nil {
			return protocol.Range{}, err
		}
		if s, err := spn.WithOffset(m.Converter); err == nil {
			start := s.Start()
			offset := start.Offset()
			if offset < len(data) {
				if width := bytes.IndexAny(data[offset:], " \n,():;[]"); width > 0 {
					spn = span.New(spn.URI(), start, span.NewPoint(start.Line(), start.Column()+width, offset+width))
				}
			}
		}
	}
	return m.Range(spn)
}

// parseGoListError attempts to parse a standard `go list` error message
// by stripping off the trailing error message.
//
// It works only on errors whose message is prefixed by colon,
// followed by a space (": "). For example:
//
//   attributes.go:13:1: expected 'package', found 'type'
//
func parseGoListError(input string) span.Span {
	input = strings.TrimSpace(input)
	msgIndex := strings.Index(input, ": ")
	if msgIndex < 0 {
		return span.Parse(input)
	}
	return span.Parse(input[:msgIndex])
}
