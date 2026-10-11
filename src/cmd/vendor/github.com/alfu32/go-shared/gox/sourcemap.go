package gox

import (
	"bytes"
	"go/scanner"
	"go/token"
)

// LineMapping describes the origin of an emitted line. Synthetic helper lines
// inherit the nearest source anchor; mappings are line-granular, not columns.
type LineMapping struct {
	GeneratedLine int
	SourceFile    string
	SourceLine    int
}

func lineMappings(filename string, src []byte) []LineMapping {
	fs := token.NewFileSet()
	file := fs.AddFile(filename, -1, len(src))
	var scan scanner.Scanner
	scan.Init(file, src, nil, scanner.ScanComments)
	for {
		_, tok, _ := scan.Scan()
		if tok == token.EOF {
			break
		}
	}
	result := []LineMapping{}
	offset := 0
	for line, b := range bytes.Split(src, []byte{'\n'}) {
		pos := file.PositionFor(file.Pos(offset), true)
		result = append(result, LineMapping{line + 1, pos.Filename, pos.Line})
		offset += len(b) + 1
	}
	return result
}
