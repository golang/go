// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"go/token"
	"net/url"

	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

// fromProtocolURI converts a protocol.DocumentURI to a source.URI.
// TODO(rstambler): Add logic here to support Windows.
func fromProtocolURI(uri protocol.DocumentURI) (source.URI, error) {
	unescaped, err := url.PathUnescape(string(uri))
	if err != nil {
		return "", err
	}
	return source.URI(unescaped), nil
}

// fromProtocolLocation converts from a protocol location to a source range.
// It will return an error if the file of the location was not valid.
// It uses fromProtocolRange to convert the start and end positions.
func fromProtocolLocation(ctx context.Context, v *cache.View, loc protocol.Location) (source.Range, error) {
	sourceURI, err := fromProtocolURI(loc.URI)
	if err != nil {
		return source.Range{}, err
	}
	f, err := v.GetFile(ctx, sourceURI)
	if err != nil {
		return source.Range{}, err
	}
	tok := f.GetToken()
	return fromProtocolRange(tok, loc.Range), nil
}

// toProtocolLocation converts from a source range back to a protocol location.
func toProtocolLocation(fset *token.FileSet, r source.Range) protocol.Location {
	tok := fset.File(r.Start)
	uri := source.ToURI(tok.Name())
	return protocol.Location{
		URI:   protocol.DocumentURI(uri),
		Range: toProtocolRange(tok, r),
	}
}

// fromProtocolRange converts a protocol range to a source range.
// It uses fromProtocolPosition to convert the start and end positions, which
// requires the token file the positions belongs to.
func fromProtocolRange(f *token.File, r protocol.Range) source.Range {
	start := fromProtocolPosition(f, r.Start)
	var end token.Pos
	switch {
	case r.End == r.Start:
		end = start
	case r.End.Line < 0:
		end = token.NoPos
	default:
		end = fromProtocolPosition(f, r.End)
	}
	return source.Range{
		Start: start,
		End:   end,
	}
}

// toProtocolRange converts from a source range back to a protocol range.
func toProtocolRange(f *token.File, r source.Range) protocol.Range {
	return protocol.Range{
		Start: toProtocolPosition(f, r.Start),
		End:   toProtocolPosition(f, r.End),
	}
}

// fromProtocolPosition converts a protocol position (0-based line and column
// number) to a token.Pos (byte offset value).
// It requires the token file the pos belongs to in order to do this.
func fromProtocolPosition(f *token.File, pos protocol.Position) token.Pos {
	line := lineStart(f, int(pos.Line)+1)
	return line + token.Pos(pos.Character) // TODO: this is wrong, bytes not characters
}

// toProtocolPosition converts from a token pos (byte offset) to a protocol
// position  (0-based line and column number)
// It requires the token file the pos belongs to in order to do this.
func toProtocolPosition(f *token.File, pos token.Pos) protocol.Position {
	if !pos.IsValid() {
		return protocol.Position{Line: -1.0, Character: -1.0}
	}
	p := f.Position(pos)
	return protocol.Position{
		Line:      float64(p.Line - 1),
		Character: float64(p.Column - 1),
	}
}

// fromTokenPosition converts a token.Position (1-based line and column
// number) to a token.Pos (byte offset value).
// It requires the token file the pos belongs to in order to do this.
func fromTokenPosition(f *token.File, pos token.Position) token.Pos {
	line := lineStart(f, pos.Line)
	return line + token.Pos(pos.Column-1) // TODO: this is wrong, bytes not characters
}

// this functionality was borrowed from the analysisutil package
func lineStart(f *token.File, line int) token.Pos {
	// Use binary search to find the start offset of this line.
	//
	// TODO(rstambler): eventually replace this function with the
	// simpler and more efficient (*go/token.File).LineStart, added
	// in go1.12.

	min := 0        // inclusive
	max := f.Size() // exclusive
	for {
		offset := (min + max) / 2
		pos := f.Pos(offset)
		posn := f.Position(pos)
		if posn.Line == line {
			return pos - (token.Pos(posn.Column) - 1)
		}

		if min+1 >= max {
			return token.NoPos
		}

		if posn.Line < line {
			min = offset
		} else {
			max = offset
		}
	}
}
