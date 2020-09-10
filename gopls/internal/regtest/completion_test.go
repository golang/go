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
	)
	for _, tc := range []struct {
		name      string
		filename  string
		content   *string
		line, col int
		want      []string
	}{
		{
			name:     "package completion at valid position",
			filename: "fruits/testfile.go",
			line:     1, col: 0,
			want: []string{"package apple", "package apple_test", "package fruits", "package fruits_test", "package main"},
		},
		{
			name:     "package completion in a comment",
			filename: "fruits/testfile.go",
			line:     0, col: 5,
			want: nil,
		},
		{
			name:     "package completion at invalid position",
			filename: "fruits/testfile.go",
			line:     4, col: 0,
			want: nil,
		},
		{
			name:     "package completion after keyword 'package'",
			filename: "fruits/testfile2.go",
			line:     0, col: 7,
			want: []string{"package apple", "package apple_test", "package fruits", "package fruits_test", "package main"},
		},
		{
			name:     "package completion with 'pac' prefix",
			filename: "fruits/testfile3.go",
			line:     0, col: 3,
			want: []string{"package apple", "package apple_test", "package fruits", "package fruits_test", "package main"},
		},
		{
			name:     "package completion at end of file",
			filename: "fruits/testfile4.go",
			line:     0, col: 0,
			content: &testfile4,
			want:    []string{"package apple", "package apple_test", "package fruits", "package fruits_test", "package main"},
		},
		{
			name:     "package completion without terminal newline",
			filename: "fruits/testfile5.go",
			line:     0, col: 14,
			content: &testfile5,
			want:    []string{"package apple", "package apple_test", "package fruits", "package fruits_test", "package main"},
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
				completions, err := env.Editor.Completion(env.Ctx, tc.filename, fake.Pos{
					Line:   tc.line,
					Column: tc.col,
				})
				if err != nil {
					t.Fatal(err)
				}
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
		completions, err := env.Editor.Completion(env.Ctx, "math/add.go", fake.Pos{
			Line:   0,
			Column: 10,
		})
		if err != nil {
			t.Fatal(err)
		}

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
