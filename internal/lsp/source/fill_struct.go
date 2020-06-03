// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/format"
	"go/types"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/lsp/protocol"
)

// FillStruct completes all of targeted struct's fields with their default values.
func FillStruct(ctx context.Context, snapshot Snapshot, fh FileHandle, protoRng protocol.Range) ([]protocol.CodeAction, error) {

	pkg, pgh, err := getParsedFile(ctx, snapshot, fh, NarrowestPackageHandle)
	if err != nil {
		return nil, fmt.Errorf("getting file for struct fill code action: %v", err)
	}
	file, src, m, _, err := pgh.Cached()
	if err != nil {
		return nil, err
	}
	spn, err := m.PointSpan(protoRng.Start)
	if err != nil {
		return nil, err
	}
	spanRng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	path, _ := astutil.PathEnclosingInterval(file, spanRng.Start, spanRng.End)
	if path == nil {
		return nil, nil
	}

	ecl := enclosingCompositeLiteral(path, spanRng.Start, pkg.GetTypesInfo())
	if ecl == nil || !ecl.isStruct() {
		return nil, nil
	}

	// If in F{ Bar<> : V} or anywhere in F{Bar : V, ...}
	// we should not fill the struct.
	if ecl.inKey || len(ecl.cl.Elts) != 0 {
		return nil, nil
	}

	var codeActions []protocol.CodeAction
	qfFunc := qualifier(file, pkg.GetTypes(), pkg.GetTypesInfo())
	switch obj := ecl.clType.(type) {
	case *types.Struct:
		fieldCount := obj.NumFields()
		if fieldCount == 0 {
			return nil, nil
		}
		var fieldSourceCode strings.Builder
		for i := 0; i < fieldCount; i++ {
			field := obj.Field(i)
			// Ignore fields that are not accessible in the current package.
			if field.Pkg() != nil && field.Pkg() != pkg.GetTypes() && !field.Exported() {
				continue
			}

			label := field.Name()
			value := formatZeroValue(field.Type(), qfFunc)
			fieldSourceCode.WriteString("\n")
			fieldSourceCode.WriteString(label)
			fieldSourceCode.WriteString(" : ")
			fieldSourceCode.WriteString(value)
			fieldSourceCode.WriteString(",")
		}

		if fieldSourceCode.Len() == 0 {
			return nil, nil
		}

		fieldSourceCode.WriteString("\n")

		// the range of all text between '<>', inclusive. E.g.  {<> ...  <}>
		mappedRange := newMappedRange(snapshot.View().Session().Cache().FileSet(), m, ecl.cl.Lbrace, ecl.cl.Rbrace+1)
		protoRange, err := mappedRange.Range()
		if err != nil {
			return nil, err
		}
		// consider formatting from the first character of the line the lbrace is on.
		// ToOffset is 1-based
		beginOffset, err := m.Converter.ToOffset(int(protoRange.Start.Line)+1, 1)
		if err != nil {
			return nil, err
		}

		endOffset, err := m.Converter.ToOffset(int(protoRange.Start.Line)+1, int(protoRange.Start.Character)+1)
		if err != nil {
			return nil, err
		}

		// An increment to make sure the lbrace is included in the slice.
		endOffset++
		// Append the edits. Then append the closing brace.
		var newSourceCode strings.Builder
		newSourceCode.Grow(endOffset - beginOffset + fieldSourceCode.Len() + 1)
		newSourceCode.WriteString(string(src[beginOffset:endOffset]))
		newSourceCode.WriteString(fieldSourceCode.String())
		newSourceCode.WriteString("}")

		buf, err := format.Source([]byte(newSourceCode.String()))
		if err != nil {
			return nil, err
		}

		// it is guaranteed that a left brace exists.
		var edit = string(buf[strings.IndexByte(string(buf), '{'):])

		codeActions = append(codeActions, protocol.CodeAction{
			Title: "Fill struct",
			Kind:  protocol.RefactorRewrite,
			Edit: protocol.WorkspaceEdit{
				DocumentChanges: []protocol.TextDocumentEdit{
					{
						TextDocument: protocol.VersionedTextDocumentIdentifier{
							Version: fh.Identity().Version,
							TextDocumentIdentifier: protocol.TextDocumentIdentifier{
								URI: protocol.URIFromSpanURI(fh.Identity().URI),
							},
						},
						Edits: []protocol.TextEdit{
							{
								Range:   protoRange,
								NewText: edit,
							},
						},
					},
				},
			},
		})
	}

	return codeActions, nil
}
