package cache

import (
	"bytes"
	"context"
	"go/scanner"
	"go/types"
	"strings"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func sourceError(ctx context.Context, view *view, pkg *pkg, e error) (*source.Error, error) {
	var (
		spn  span.Span
		msg  string
		kind packages.ErrorKind
	)
	switch e := e.(type) {
	case packages.Error:
		if e.Pos == "" {
			spn = parseGoListError(e.Msg)
		} else {
			spn = span.Parse(e.Pos)
		}
		msg = e.Msg
		kind = e.Kind
	case *scanner.Error:
		msg = e.Msg
		kind = packages.ParseError
		spn = span.Parse(e.Pos.String())
	case scanner.ErrorList:
		// The first parser error is likely the root cause of the problem.
		if e.Len() > 0 {
			spn = span.Parse(e[0].Pos.String())
			msg = e[0].Msg
			kind = packages.ParseError
		}
	case types.Error:
		spn = span.Parse(view.session.cache.fset.Position(e.Pos).String())
		msg = e.Msg
		kind = packages.TypeError
	}
	rng, err := spanToRange(ctx, pkg, spn, kind == packages.TypeError)
	if err != nil {
		return nil, err
	}
	return &source.Error{
		URI:   spn.URI(),
		Range: rng,
		Msg:   msg,
		Kind:  kind,
	}, nil
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
	data, _, err := ph.File().Read(ctx)
	if err != nil {
		return protocol.Range{}, err
	}
	if spn.IsPoint() && isTypeError {
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
