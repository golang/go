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
	token.Position
	Message string
}

func Diagnostics(ctx context.Context, f File) (map[string][]Diagnostic, error) {
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
		pos := errorPos(diag)
		diagnostic := Diagnostic{
			Position: pos,
			Message:  diag.Msg,
		}
		if _, ok := reports[pos.Filename]; ok {
			reports[pos.Filename] = append(reports[pos.Filename], diagnostic)
		}
	}
	return reports, nil
}

// FromTokenPosition converts a token.Position (1-based line and column
// number) to a token.Pos (byte offset value).
// It requires the token file the pos belongs to in order to do this.
func FromTokenPosition(f *token.File, pos token.Position) token.Pos {
	line := lineStart(f, pos.Line)
	return line + token.Pos(pos.Column-1) // TODO: this is wrong, bytes not characters
}

func errorPos(pkgErr packages.Error) token.Position {
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
	return pos
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
