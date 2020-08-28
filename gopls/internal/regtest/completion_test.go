// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"fmt"
	"testing"

	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
)

func TestPackageCompletion(t *testing.T) {
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

-- fruits/testfile4.go --`
	for _, testcase := range []struct {
		name      string
		filename  string
		line, col int
		want      []string
	}{
		{
			"package completion at valid position",
			"fruits/testfile.go", 1, 0,
			[]string{"package apple", "package apple_test", "package fruits", "package fruits_test", "package main"},
		},
		{
			"package completion in a comment",
			"fruits/testfile.go", 0, 5,
			nil,
		},
		{
			"package completion at invalid position",
			"fruits/testfile.go", 4, 0,
			nil,
		},
		{
			"package completion works after keyword 'package'",
			"fruits/testfile2.go", 0, 7,
			[]string{"package apple", "package apple_test", "package fruits", "package fruits_test", "package main"},
		},
		{
			"package completion works with a prefix for keyword 'package'",
			"fruits/testfile3.go", 0, 3,
			[]string{"package apple", "package apple_test", "package fruits", "package fruits_test", "package main"},
		},
		{
			"package completion at end of file",
			"fruits/testfile4.go", 0, 0,
			[]string{"package apple", "package apple_test", "package fruits", "package fruits_test", "package main"},
		},
	} {
		t.Run(testcase.name, func(t *testing.T) {
			run(t, files, func(t *testing.T, env *Env) {
				env.OpenFile(testcase.filename)
				completions, err := env.Editor.Completion(env.Ctx, testcase.filename, fake.Pos{
					Line:   testcase.line,
					Column: testcase.col,
				})
				if err != nil {
					t.Fatal(err)
				}
				diff := compareCompletionResults(testcase.want, completions.Items)
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
