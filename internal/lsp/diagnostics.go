// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"fmt"
	"go/token"
	"strconv"
	"strings"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func diagnostics(v *source.View, uri source.URI) (map[string][]protocol.Diagnostic, error) {
	pkg, err := v.GetFile(uri).GetPackage()
	if err != nil {
		return nil, err
	}
	if pkg == nil {
		return nil, fmt.Errorf("package for %v not found", uri)
	}
	reports := make(map[string][]protocol.Diagnostic)
	for _, filename := range pkg.GoFiles {
		reports[filename] = []protocol.Diagnostic{}
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
	errors := typeErrors
	if len(parseErrors) > 0 {
		errors = parseErrors
	}
	for _, err := range errors {
		pos := parseErrorPos(err)
		line := float64(pos.Line) - 1
		col := float64(pos.Column) - 1
		diagnostic := protocol.Diagnostic{
			// TODO(rstambler): Add support for diagnostic ranges.
			Range: protocol.Range{
				Start: protocol.Position{
					Line:      line,
					Character: col,
				},
				End: protocol.Position{
					Line:      line,
					Character: col,
				},
			},
			Severity: protocol.SeverityError,
			Source:   "LSP: Go compiler",
			Message:  err.Msg,
		}
		if _, ok := reports[pos.Filename]; ok {
			reports[pos.Filename] = append(reports[pos.Filename], diagnostic)
		}
	}
	return reports, nil
}

func parseErrorPos(pkgErr packages.Error) (pos token.Position) {
	remainder1, first, hasLine := chop(pkgErr.Pos)
	remainder2, second, hasColumn := chop(remainder1)
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
