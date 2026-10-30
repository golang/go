// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/analysis/analyzerutil"
)

var importCommentAnalyzer = &analysis.Analyzer{
	Name: "importcomment",
	Doc:  analyzerutil.MustExtractDoc(doc, "importcomment"),
	URL:  "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#hdr-Analyzer_importcomment",
	Run:  importcomment,
}

func importcomment(pass *analysis.Pass) (any, error) {
	// Import path comments are ignored in module mode.
	if pass.Module == nil {
		return nil, nil
	}

	for _, file := range pass.Files {
		// An import comment follows the package name on the same line.
		pkgEnd := file.Name.End()
		pkgLine := pass.Fset.Position(pkgEnd).Line
		for _, c := range file.Comments {
			if len(c.List) != 1 {
				continue
			}
			if c.Pos() < pkgEnd {
				continue
			}
			commentLine := pass.Fset.Position(c.Pos()).Line
			if commentLine > pkgLine {
				break // comments are sorted; the rest are on later lines
			}
			// Have: package p // comment
			if !isImportComment(c.Text()) {
				continue
			}
			pass.Report(analysis.Diagnostic{
				Pos:     c.Pos(),
				End:     c.End(),
				Message: "canonical import path comment is ignored in module mode",
				SuggestedFixes: []analysis.SuggestedFix{{
					Message: "Remove obsolete import path comment",
					TextEdits: []analysis.TextEdit{{
						Pos: pkgEnd, // deletes the preceding space too
						End: c.End(),
					}},
				}},
			})
		}
	}

	return nil, nil
}

// isImportComment reports whether text, a comment's content with its
// markers removed, is a canonical import path comment, import "path".
func isImportComment(text string) bool {
	text = strings.TrimSpace(text)
	return strings.HasPrefix(text, `import "`) && strings.HasSuffix(text, `"`)
}
