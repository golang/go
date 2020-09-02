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

-- fruits/testfile.go --`

	want := []string{"package apple", "package apple_test", "package fruits", "package fruits_test", "package main"}
	run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("fruits/testfile.go")
		content := env.ReadWorkspaceFile("fruits/testfile.go")
		if content != "" {
			t.Fatal("testfile.go should be empty to test completion on end of file without newline")
		}

		completions, err := env.Editor.Completion(env.Ctx, "fruits/testfile.go", fake.Pos{
			Line:   0,
			Column: 0,
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
