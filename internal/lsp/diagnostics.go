package lsp

import (
	"go/token"
	"strconv"
	"strings"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/protocol"
)

func (v *view) diagnostics(uri protocol.DocumentURI) (map[string][]protocol.Diagnostic, error) {
	pkg, err := v.typeCheck(uri)
	if err != nil {
		return nil, err
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
	split := strings.Split(pkgErr.Pos, ":")
	if len(split) <= 1 {
		return pos
	}
	pos.Filename = split[0]
	line, err := strconv.ParseInt(split[1], 10, 64)
	if err != nil {
		return pos
	}
	pos.Line = int(line)
	if len(split) == 3 {
		col, err := strconv.ParseInt(split[2], 10, 64)
		if err != nil {
			return pos
		}
		pos.Column = int(col)
	}
	return pos

}
