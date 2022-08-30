// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"context"
	"fmt"
	"strings"
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"

	"golang.org/x/tools/gopls/internal/lsp/fake"
)

type completionBenchOptions struct {
	file, locationRegexp string

	// Hooks to run edits before initial completion
	setup            func(*Env) // run before the benchmark starts
	beforeCompletion func(*Env) // run before each completion
}

func benchmarkCompletion(options completionBenchOptions, b *testing.B) {
	dir := benchmarkDir()

	// Use a new environment for each test, to avoid any existing state from the
	// previous session.
	sandbox, editor, awaiter, err := connectEditor(dir, fake.EditorConfig{
		Settings: map[string]interface{}{
			"completionBudget": "1m", // arbitrary long completion budget
		},
	})
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	defer func() {
		if err := editor.Close(ctx); err != nil {
			b.Errorf("closing editor: %v", err)
		}
	}()

	env := &Env{
		T:       b,
		Ctx:     ctx,
		Editor:  editor,
		Sandbox: sandbox,
		Awaiter: awaiter,
	}

	// Run edits required for this completion.
	if options.setup != nil {
		options.setup(env)
	}

	// Run a completion to make sure the system is warm.
	pos := env.RegexpSearch(options.file, options.locationRegexp)
	completions := env.Completion(options.file, pos)

	if testing.Verbose() {
		fmt.Println("Results:")
		for i := 0; i < len(completions.Items); i++ {
			fmt.Printf("\t%d. %v\n", i, completions.Items[i])
		}
	}

	b.ResetTimer()

	// Use a subtest to ensure that benchmarkCompletion does not itself get
	// executed multiple times (as it is doing expensive environment
	// initialization).
	b.Run("completion", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if options.beforeCompletion != nil {
				options.beforeCompletion(env)
			}
			env.Completion(options.file, pos)
		}
	})
}

// endPosInBuffer returns the position for last character in the buffer for
// the given file.
func endPosInBuffer(env *Env, name string) fake.Pos {
	buffer := env.Editor.BufferText(name)
	lines := strings.Split(buffer, "\n")
	numLines := len(lines)

	return fake.Pos{
		Line:   numLines - 1,
		Column: len([]rune(lines[numLines-1])),
	}
}

// Benchmark struct completion in tools codebase.
func BenchmarkStructCompletion(b *testing.B) {
	file := "internal/lsp/cache/session.go"

	setup := func(env *Env) {
		env.OpenFile(file)
		originalBuffer := env.Editor.BufferText(file)
		env.EditBuffer(file, fake.Edit{
			End:  endPosInBuffer(env, file),
			Text: originalBuffer + "\nvar testVariable map[string]bool = Session{}.\n",
		})
	}

	benchmarkCompletion(completionBenchOptions{
		file:           file,
		locationRegexp: `var testVariable map\[string\]bool = Session{}(\.)`,
		setup:          setup,
	}, b)
}

// Benchmark import completion in tools codebase.
func BenchmarkImportCompletion(b *testing.B) {
	const file = "internal/lsp/source/completion/completion.go"
	benchmarkCompletion(completionBenchOptions{
		file:           file,
		locationRegexp: `go\/()`,
		setup:          func(env *Env) { env.OpenFile(file) },
	}, b)
}

// Benchmark slice completion in tools codebase.
func BenchmarkSliceCompletion(b *testing.B) {
	file := "internal/lsp/cache/session.go"

	setup := func(env *Env) {
		env.OpenFile(file)
		originalBuffer := env.Editor.BufferText(file)
		env.EditBuffer(file, fake.Edit{
			End:  endPosInBuffer(env, file),
			Text: originalBuffer + "\nvar testVariable []byte = \n",
		})
	}

	benchmarkCompletion(completionBenchOptions{
		file:           file,
		locationRegexp: `var testVariable \[\]byte (=)`,
		setup:          setup,
	}, b)
}

// Benchmark deep completion in function call in tools codebase.
func BenchmarkFuncDeepCompletion(b *testing.B) {
	file := "internal/lsp/source/completion/completion.go"
	fileContent := `
func (c *completer) _() {
	c.inference.kindMatches(c.)
}
`
	setup := func(env *Env) {
		env.OpenFile(file)
		originalBuffer := env.Editor.BufferText(file)
		env.EditBuffer(file, fake.Edit{
			End:  endPosInBuffer(env, file),
			Text: originalBuffer + fileContent,
		})
	}

	benchmarkCompletion(completionBenchOptions{
		file:           file,
		locationRegexp: `func \(c \*completer\) _\(\) {\n\tc\.inference\.kindMatches\((c)`,
		setup:          setup,
	}, b)
}

// Benchmark completion following an arbitrary edit.
//
// Edits force type-checked packages to be invalidated, so we want to measure
// how long it takes before completion results are available.
func BenchmarkCompletionFollowingEdit(b *testing.B) {
	file := "internal/lsp/source/completion/completion2.go"
	fileContent := `
package completion

func (c *completer) _() {
	c.inference.kindMatches(c.)
	// __MAGIC_STRING_1
}
`
	setup := func(env *Env) {
		env.CreateBuffer(file, fileContent)
	}

	n := 1
	beforeCompletion := func(env *Env) {
		old := fmt.Sprintf("__MAGIC_STRING_%d", n)
		new := fmt.Sprintf("__MAGIC_STRING_%d", n+1)
		n++
		env.RegexpReplace(file, old, new)
	}

	benchmarkCompletion(completionBenchOptions{
		file:             file,
		locationRegexp:   `func \(c \*completer\) _\(\) {\n\tc\.inference\.kindMatches\((c)`,
		setup:            setup,
		beforeCompletion: beforeCompletion,
	}, b)
}
