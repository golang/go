// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"fmt"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/testenv"
)

func TestPackageCompletion(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)
	const files = `
-- go.mod --
module mod.com

-- fruits/apple.go --
package apple

fun apple() int {
	return 0
}

-- fruits/testfile.go --
// this is a comment

/*
 this is a multiline comment
*/

import "fmt"

func test() {}

-- fruits/testfile2.go --
package

-- fruits/testfile3.go --
pac
`
	var (
		testfile4 = ""
		testfile5 = "/*a comment*/ "
		testfile6 = "/*a comment*/\n"
	)
	for _, tc := range []struct {
		name          string
		filename      string
		content       *string
		triggerRegexp string
		want          []string
		editRegexp    string
	}{
		{
			name:          "package completion at valid position",
			filename:      "fruits/testfile.go",
			triggerRegexp: "\n()",
			want:          []string{"package apple", "package apple_test", "package fruits", "package fruits_test", "package main"},
			editRegexp:    "\n()",
		},
		{
			name:          "package completion in a comment",
			filename:      "fruits/testfile.go",
			triggerRegexp: "th(i)s",
			want:          nil,
		},
		{
			name:          "package completion in a multiline comment",
			filename:      "fruits/testfile.go",
			triggerRegexp: `\/\*\n()`,
			want:          nil,
		},
		{
			name:          "package completion at invalid position",
			filename:      "fruits/testfile.go",
			triggerRegexp: "import \"fmt\"\n()",
			want:          nil,
		},
		{
			name:          "package completion after keyword 'package'",
			filename:      "fruits/testfile2.go",
			triggerRegexp: "package()",
			want:          []string{"package apple", "package apple_test", "package fruits", "package fruits_test", "package main"},
			editRegexp:    "package\n",
		},
		{
			name:          "package completion with 'pac' prefix",
			filename:      "fruits/testfile3.go",
			triggerRegexp: "pac()",
			want:          []string{"package apple", "package apple_test", "package fruits", "package fruits_test", "package main"},
			editRegexp:    "pac",
		},
		{
			name:          "package completion for empty file",
			filename:      "fruits/testfile4.go",
			triggerRegexp: "^$",
			content:       &testfile4,
			want:          []string{"package apple", "package apple_test", "package fruits", "package fruits_test", "package main"},
			editRegexp:    "^$",
		},
		{
			name:          "package completion without terminal newline",
			filename:      "fruits/testfile5.go",
			triggerRegexp: `\*\/ ()`,
			content:       &testfile5,
			want:          []string{"package apple", "package apple_test", "package fruits", "package fruits_test", "package main"},
			editRegexp:    `\*\/ ()`,
		},
		{
			name:          "package completion on terminal newline",
			filename:      "fruits/testfile6.go",
			triggerRegexp: `\*\/\n()`,
			content:       &testfile6,
			want:          []string{"package apple", "package apple_test", "package fruits", "package fruits_test", "package main"},
			editRegexp:    `\*\/\n()`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			run(t, files, func(t *testing.T, env *Env) {
				if tc.content != nil {
					env.WriteWorkspaceFile(tc.filename, *tc.content)
					env.Await(
						CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChangeWatchedFiles), 1),
					)
				}
				env.OpenFile(tc.filename)
				completions := env.Completion(tc.filename, env.RegexpSearch(tc.filename, tc.triggerRegexp))

				// Check that the completion item suggestions are in the range
				// of the file.
				lineCount := len(strings.Split(env.Editor.BufferText(tc.filename), "\n"))
				for _, item := range completions.Items {
					if start := int(item.TextEdit.Range.Start.Line); start >= lineCount {
						t.Fatalf("unexpected text edit range start line number: got %d, want less than %d", start, lineCount)
					}
					if end := int(item.TextEdit.Range.End.Line); end >= lineCount {
						t.Fatalf("unexpected text edit range end line number: got %d, want less than %d", end, lineCount)
					}
				}

				if tc.want != nil {
					start, end := env.RegexpRange(tc.filename, tc.editRegexp)
					expectedRng := protocol.Range{
						Start: fake.Pos.ToProtocolPosition(start),
						End:   fake.Pos.ToProtocolPosition(end),
					}
					for _, item := range completions.Items {
						gotRng := item.TextEdit.Range
						if expectedRng != gotRng {
							t.Errorf("unexpected completion range for completion item %s: got %v, want %v",
								item.Label, gotRng, expectedRng)
						}
					}
				}

				diff := compareCompletionResults(tc.want, completions.Items)
				if diff != "" {
					t.Error(diff)
				}
			})
		})
	}
}

func TestPackageNameCompletion(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

-- math/add.go --
package ma
`

	want := []string{"ma", "ma_test", "main", "math", "math_test"}
	run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("math/add.go")
		completions := env.Completion("math/add.go", fake.Pos{
			Line:   0,
			Column: 10,
		})

		diff := compareCompletionResults(want, completions.Items)
		if diff != "" {
			t.Fatal(diff)
		}
	})
}

func compareCompletionResults(want []string, gotItems []protocol.CompletionItem) string {
	if len(gotItems) != len(want) {
		return fmt.Sprintf("got %v completion(s), want %v", len(gotItems), len(want))
	}

	var got []string
	for _, item := range gotItems {
		got = append(got, item.Label)
	}

	for i, v := range got {
		if v != want[i] {
			return fmt.Sprintf("completion results are not the same: got %v, want %v", got, want)
		}
	}

	return ""
}
