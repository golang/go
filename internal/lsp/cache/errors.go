// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"context"
	"go/scanner"
	"go/token"
	"go/types"
	"regexp"
	"strconv"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	errors "golang.org/x/xerrors"
)

func sourceError(ctx context.Context, fset *token.FileSet, pkg *pkg, e interface{}) (*source.Error, error) {
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
		kind = toSourceErrorKind(e.Kind)
		var ok bool
		if msg, spn, ok = parseGoListImportCycleError(ctx, fset, e, pkg); ok {
			kind = source.TypeError
			break
		}
		if e.Pos == "" {
			spn = parseGoListError(e.Msg)

			// We may not have been able to parse a valid span.
			if _, err := spanToRange(ctx, pkg, spn); err != nil {
				return &source.Error{
					URI:     spn.URI(),
					Message: msg,
					Kind:    kind,
				}, nil
			}
		} else {
			spn = span.Parse(e.Pos)
		}
	case *scanner.Error:
		msg = e.Msg
		kind = source.ParseError
		spn, err = scannerErrorRange(ctx, fset, pkg, e.Pos)
		if err != nil {
			log.Error(ctx, "no span for scanner.Error pos", err, telemetry.Package.Of(pkg.ID()))
			spn = span.Parse(e.Pos.String())
		}

	case scanner.ErrorList:
		// The first parser error is likely the root cause of the problem.
		if e.Len() <= 0 {
			return nil, errors.Errorf("no errors in %v", e)
		}
		msg = e[0].Msg
		kind = source.ParseError
		spn, err = scannerErrorRange(ctx, fset, pkg, e[0].Pos)
		if err != nil {
			log.Error(ctx, "no span for scanner.Error pos", err, telemetry.Package.Of(pkg.ID()))
			spn = span.Parse(e[0].Pos.String())
		}
	case types.Error:
		msg = e.Msg
		kind = source.TypeError
		spn, err = typeErrorRange(ctx, fset, pkg, e.Pos)
		if err != nil {
			return nil, err
		}

	case *analysis.Diagnostic:
		spn, err = span.NewRange(fset, e.Pos, e.End).Span()
		if err != nil {
			return nil, err
		}
		msg = e.Message
		kind = source.Analysis
		category = e.Category
		fixes, err = suggestedFixes(ctx, fset, pkg, e)
		if err != nil {
			return nil, err
		}
		related, err = relatedInformation(ctx, fset, pkg, e)
		if err != nil {
			return nil, err
		}
	}
	rng, err := spanToRange(ctx, pkg, spn)
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

func suggestedFixes(ctx context.Context, fset *token.FileSet, pkg *pkg, diag *analysis.Diagnostic) ([]source.SuggestedFix, error) {
	var fixes []source.SuggestedFix
	for _, fix := range diag.SuggestedFixes {
		edits := make(map[span.URI][]protocol.TextEdit)
		for _, e := range fix.TextEdits {
			spn, err := span.NewRange(fset, e.Pos, e.End).Span()
			if err != nil {
				return nil, err
			}
			rng, err := spanToRange(ctx, pkg, spn)
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

func relatedInformation(ctx context.Context, fset *token.FileSet, pkg *pkg, diag *analysis.Diagnostic) ([]source.RelatedInformation, error) {
	var out []source.RelatedInformation
	for _, related := range diag.Related {
		spn, err := span.NewRange(fset, related.Pos, related.End).Span()
		if err != nil {
			return nil, err
		}
		rng, err := spanToRange(ctx, pkg, spn)
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

func typeErrorRange(ctx context.Context, fset *token.FileSet, pkg *pkg, pos token.Pos) (span.Span, error) {
	posn := fset.Position(pos)
	ph, _, err := source.FindFileInPackage(pkg, span.URIFromPath(posn.Filename))
	if err != nil {
		return span.Span{}, err
	}
	_, _, m, _, err := ph.Cached()
	if err != nil {
		return span.Span{}, err
	}
	spn, err := span.Range{
		FileSet:   fset,
		Start:     pos,
		End:       pos,
		Converter: m.Converter,
	}.Span()
	if err != nil {
		return span.Span{}, err
	}
	s, err := spn.WithOffset(m.Converter)
	if err != nil {
		return span.Span{}, err
	}
	data, _, err := ph.File().Read(ctx)
	if err != nil {
		return span.Span{}, err
	}
	start := s.Start()
	offset := start.Offset()
	if offset < len(data) {
		if width := bytes.IndexAny(data[offset:], " \n,():;[]"); width > 0 {
			return span.New(spn.URI(), start, span.NewPoint(start.Line(), start.Column()+width, offset+width)), nil
		}
	}
	return spn, nil
}

func scannerErrorRange(ctx context.Context, fset *token.FileSet, pkg *pkg, posn token.Position) (span.Span, error) {
	ph, _, err := source.FindFileInPackage(pkg, span.URIFromPath(posn.Filename))
	if err != nil {
		return span.Span{}, err
	}
	file, _, _, _, err := ph.Cached()
	if err != nil {
		return span.Span{}, err
	}
	tok := fset.File(file.Pos())
	if tok == nil {
		return span.Span{}, errors.Errorf("no token.File for %s", ph.File().Identity().URI)
	}
	pos := tok.Pos(posn.Offset)
	return span.NewRange(fset, pos, pos).Span()
}

// spanToRange converts a span.Span to a protocol.Range,
// assuming that the span belongs to the package whose diagnostics are being computed.
func spanToRange(ctx context.Context, pkg *pkg, spn span.Span) (protocol.Range, error) {
	ph, _, err := source.FindFileInPackage(pkg, spn.URI())
	if err != nil {
		return protocol.Range{}, err
	}
	_, _, m, _, err := ph.Cached()
	if err != nil {
		return protocol.Range{}, err
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

func parseGoListImportCycleError(ctx context.Context, fset *token.FileSet, e packages.Error, pkg *pkg) (string, span.Span, bool) {
	re := regexp.MustCompile(`(.*): import stack: \[(.+)\]`)
	matches := re.FindStringSubmatch(strings.TrimSpace(e.Msg))
	if len(matches) < 3 {
		return e.Msg, span.Span{}, false
	}
	msg := matches[1]
	importList := strings.Split(matches[2], " ")
	// Since the error is relative to the current package. The import that is causing
	// the import cycle error is the second one in the list.
	if len(importList) < 2 {
		return msg, span.Span{}, false
	}
	// Imports have quotation marks around them.
	circImp := strconv.Quote(importList[1])
	for _, ph := range pkg.compiledGoFiles {
		fh, _, _, _, err := ph.Parse(ctx)
		if err != nil {
			continue
		}
		// Search file imports for the import that is causing the import cycle.
		for _, imp := range fh.Imports {
			if imp.Path.Value == circImp {
				spn, err := span.NewRange(fset, imp.Pos(), imp.End()).Span()
				if err != nil {
					return msg, span.Span{}, false
				}
				return msg, spn, true
			}
		}
	}
	return msg, span.Span{}, false
}
