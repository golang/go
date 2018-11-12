// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/token"
	"strconv"
	"strings"

	"golang.org/x/tools/go/packages"
)

type Diagnostic struct {
	Range    Range
	Severity DiagnosticSeverity
	Message  string
}

type DiagnosticSeverity int

const (
	SeverityError DiagnosticSeverity = iota
	SeverityWarning
	SeverityHint
	SeverityInformation
)

func Diagnostics(ctx context.Context, v *View, f *File) (map[string][]Diagnostic, error) {
	pkg, err := f.GetPackage()
	if err != nil {
		return nil, err
	}
	// Prepare the reports we will send for this package.
	reports := make(map[string][]Diagnostic)
	for _, filename := range pkg.GoFiles {
		reports[filename] = []Diagnostic{}
	}
	var parseErrors, typeErrors []packages.Error
	for _, err := range pkg.Errors {
		switch err.Kind {
		case packages.ParseError:
			parseErrors = append(parseErrors, err)
		case packages.TypeError:
			typeErrors = append(typeErrors, err)
		default:
			// ignore other types of errors
			continue
		}
	}
	// Don't report type errors if there are parse errors.
	diags := typeErrors
	if len(parseErrors) > 0 {
		diags = parseErrors
	}
	for _, diag := range diags {
		filename, start := v.errorPos(diag)
		// TODO(rstambler): Add support for diagnostic ranges.
		end := start
		diagnostic := Diagnostic{
			Range: Range{
				Start: start,
				End:   end,
			},
			Message:  diag.Msg,
			Severity: SeverityError,
		}
		if _, ok := reports[filename]; ok {
			reports[filename] = append(reports[filename], diagnostic)
		}
	}
	return reports, nil
}

func (v *View) errorPos(pkgErr packages.Error) (string, token.Pos) {
	remainder1, first, hasLine := chop(pkgErr.Pos)
	remainder2, second, hasColumn := chop(remainder1)
	var pos token.Position
	if hasLine && hasColumn {
		pos.Filename = remainder2
		pos.Line = second
		pos.Column = first
	} else if hasLine {
		pos.Filename = remainder1
		pos.Line = first
	}
	f := v.GetFile(ToURI(pos.Filename))
	if f == nil {
		return "", token.NoPos
	}
	tok, err := f.GetToken()
	if err != nil {
		return "", token.NoPos
	}
	return pos.Filename, fromTokenPosition(tok, pos)
}

func chop(text string) (remainder string, value int, ok bool) {
	i := strings.LastIndex(text, ":")
	if i < 0 {
		return text, 0, false
	}
	v, err := strconv.ParseInt(text[i+1:], 10, 64)
	if err != nil {
		return text, 0, false
	}
	return text[:i], int(v), true
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
	// TODO(adonovan): eventually replace this function with the
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
