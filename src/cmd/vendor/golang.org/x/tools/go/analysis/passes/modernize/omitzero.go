// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"go/ast"
	"go/types"
	"reflect"
	"strconv"
	"strings"
	"sync"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/versions"
)

var OmitZeroAnalyzer = &analysis.Analyzer{
	Name:     "omitzero",
	Doc:      analyzerutil.MustExtractDoc(doc, "omitzero"),
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      omitzero,
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#omitzero",
}

// The omitzero pass searches for instances of "omitempty" in a json field tag on a
// struct. Since "omitfilesUsingGoVersions not have any effect when applied to a struct field,
// it suggests either deleting "omitempty" or replacing it with "omitzero", which
// correctly excludes structs from a json encoding.
func omitzero(pass *analysis.Pass) (any, error) {
	// usesKubebuilder reports whether "+kubebuilder:" appears in
	// any comment in the package, since it has its own
	// interpretation of what omitzero means; see go.dev/issue/76649.
	// It is computed once, on demand.
	usesKubebuilder := sync.OnceValue[bool](func() bool {
		for _, file := range pass.Files {
			for _, comment := range file.Comments {
				if strings.Contains(comment.Text(), "+kubebuilder:") {
					return true
				}
			}
		}
		return false
	})

	checkField := func(field *ast.Field) {
		typ := pass.TypesInfo.TypeOf(field.Type)
		_, ok := typ.Underlying().(*types.Struct)
		if !ok {
			// Not a struct
			return
		}
		tag := field.Tag
		if tag == nil {
			// No tag to check
			return
		}
		// The omitempty tag may be used by other packages besides json, but we should only modify its use with json
		tagconv, _ := strconv.Unquote(tag.Value)
		match := omitemptyRegex.FindStringSubmatchIndex(tagconv)
		if match == nil {
			// No omitempty in json tag
			return
		}
		omitEmpty, err := astutil.RangeInStringLiteral(field.Tag, match[2], match[3])
		if err != nil {
			return
		}
		var remove analysis.Range = omitEmpty

		jsonTag := reflect.StructTag(tagconv).Get("json")
		if jsonTag == ",omitempty" {
			// Remove the entire struct tag if json is the only package used
			if match[1]-match[0] == len(tagconv) {
				remove = field.Tag
			} else {
				// Remove the json tag if omitempty is the only field
				remove, err = astutil.RangeInStringLiteral(field.Tag, match[0], match[1])
				if err != nil {
					return
				}
			}
		}

		// Don't offer a fix if the package seems to use kubebuilder,
		// as it has its own intepretation of "omitzero" tags.
		// https://book.kubebuilder.io/reference/markers.html
		if usesKubebuilder() {
			return
		}

		pass.Report(analysis.Diagnostic{
			Pos:     field.Tag.Pos(),
			End:     field.Tag.End(),
			Message: "Omitempty has no effect on nested struct fields",
			SuggestedFixes: []analysis.SuggestedFix{
				{
					Message: "Remove redundant omitempty tag",
					TextEdits: []analysis.TextEdit{
						{
							Pos: remove.Pos(),
							End: remove.End(),
						},
					},
				},
				{
					Message: "Replace omitempty with omitzero (behavior change)",
					TextEdits: []analysis.TextEdit{
						{
							Pos:     omitEmpty.Pos(),
							End:     omitEmpty.End(),
							NewText: []byte(",omitzero"),
						},
					},
				},
			}})
	}

	for curFile := range filesUsingGoVersion(pass, versions.Go1_24) {
		for curStruct := range curFile.Preorder((*ast.StructType)(nil)) {
			for _, curField := range curStruct.Node().(*ast.StructType).Fields.List {
				checkField(curField)
			}
		}
	}

	return nil, nil
}
