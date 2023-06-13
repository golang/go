// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package completion

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/hooks"
	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/internal/testenv"
)

func TestMain(m *testing.M) {
	bug.PanicOnBugs = true
	Main(m, hooks.Options)
}

const proxy = `
-- example.com@v1.2.3/go.mod --
module example.com

go 1.12
-- example.com@v1.2.3/blah/blah.go --
package blah

const Name = "Blah"
-- random.org@v1.2.3/go.mod --
module random.org

go 1.12
-- random.org@v1.2.3/blah/blah.go --
package hello

const Name = "Hello"
`

func TestPackageCompletion(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.12
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
-- 123f_r.u~its-123/testfile.go --
package

-- .invalid-dir@-name/testfile.go --
package
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
		// Issue golang/go#44680
		{
			name:          "package completion for dir name with punctuation",
			filename:      "123f_r.u~its-123/testfile.go",
			triggerRegexp: "package()",
			want:          []string{"package fruits123", "package fruits123_test", "package main"},
			editRegexp:    "package\n",
		},
		{
			name:          "package completion for invalid dir name",
			filename:      ".invalid-dir@-name/testfile.go",
			triggerRegexp: "package()",
			want:          []string{"package main"},
			editRegexp:    "package\n",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			Run(t, files, func(t *testing.T, env *Env) {
				if tc.content != nil {
					env.WriteWorkspaceFile(tc.filename, *tc.content)
					env.Await(env.DoneWithChangeWatchedFiles())
				}
				env.OpenFile(tc.filename)
				completions := env.Completion(env.RegexpSearch(tc.filename, tc.triggerRegexp))

				// Check that the completion item suggestions are in the range
				// of the file. {Start,End}.Line are zero-based.
				lineCount := len(strings.Split(env.BufferText(tc.filename), "\n"))
				for _, item := range completions.Items {
					if start := int(item.TextEdit.Range.Start.Line); start > lineCount {
						t.Fatalf("unexpected text edit range start line number: got %d, want <= %d", start, lineCount)
					}
					if end := int(item.TextEdit.Range.End.Line); end > lineCount {
						t.Fatalf("unexpected text edit range end line number: got %d, want <= %d", end, lineCount)
					}
				}

				if tc.want != nil {
					expectedLoc := env.RegexpSearch(tc.filename, tc.editRegexp)
					for _, item := range completions.Items {
						gotRng := item.TextEdit.Range
						if expectedLoc.Range != gotRng {
							t.Errorf("unexpected completion range for completion item %s: got %v, want %v",
								item.Label, gotRng, expectedLoc.Range)
						}
					}
				}

				diff := compareCompletionLabels(tc.want, completions.Items)
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

go 1.12
-- math/add.go --
package ma
`

	want := []string{"ma", "ma_test", "main", "math", "math_test"}
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("math/add.go")
		completions := env.Completion(env.RegexpSearch("math/add.go", "package ma()"))

		diff := compareCompletionLabels(want, completions.Items)
		if diff != "" {
			t.Fatal(diff)
		}
	})
}

// TODO(rfindley): audit/clean up call sites for this helper, to ensure
// consistent test errors.
func compareCompletionLabels(want []string, gotItems []protocol.CompletionItem) string {
	var got []string
	for _, item := range gotItems {
		got = append(got, item.Label)
		if item.Label != item.InsertText && item.TextEdit == nil {
			// Label should be the same as InsertText, if InsertText is to be used
			return fmt.Sprintf("label not the same as InsertText %#v", item)
		}
	}

	if len(got) == 0 && len(want) == 0 {
		return "" // treat nil and the empty slice as equivalent
	}

	if diff := cmp.Diff(want, got); diff != "" {
		return fmt.Sprintf("completion item mismatch (-want +got):\n%s", diff)
	}
	return ""
}

func TestUnimportedCompletion(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.14

require example.com v1.2.3
-- go.sum --
example.com v1.2.3 h1:ihBTGWGjTU3V4ZJ9OmHITkU9WQ4lGdQkMjgyLFk0FaY=
example.com v1.2.3/go.mod h1:Y2Rc5rVWjWur0h3pd9aEvK5Pof8YKDANh9gHA2Maujo=
-- main.go --
package main

func main() {
	_ = blah
}
-- main2.go --
package main

import "example.com/blah"

func _() {
	_ = blah.Hello
}
`
	WithOptions(
		ProxyFiles(proxy),
	).Run(t, mod, func(t *testing.T, env *Env) {
		// Make sure the dependency is in the module cache and accessible for
		// unimported completions, and then remove it before proceeding.
		env.RemoveWorkspaceFile("main2.go")
		env.RunGoCommand("mod", "tidy")
		env.Await(env.DoneWithChangeWatchedFiles())

		// Trigger unimported completions for the example.com/blah package.
		env.OpenFile("main.go")
		env.Await(env.DoneWithOpen())
		loc := env.RegexpSearch("main.go", "ah")
		completions := env.Completion(loc)
		if len(completions.Items) == 0 {
			t.Fatalf("no completion items")
		}
		env.AcceptCompletion(loc, completions.Items[0]) // adds blah import to main.go
		env.Await(env.DoneWithChange())

		// Trigger completions once again for the blah.<> selector.
		env.RegexpReplace("main.go", "_ = blah", "_ = blah.")
		env.Await(env.DoneWithChange())
		loc = env.RegexpSearch("main.go", "\n}")
		completions = env.Completion(loc)
		if len(completions.Items) != 1 {
			t.Fatalf("expected 1 completion item, got %v", len(completions.Items))
		}
		item := completions.Items[0]
		if item.Label != "Name" {
			t.Fatalf("expected completion item blah.Name, got %v", item.Label)
		}
		env.AcceptCompletion(loc, item)

		// Await the diagnostics to add example.com/blah to the go.mod file.
		env.AfterChange(
			Diagnostics(env.AtRegexp("main.go", `"example.com/blah"`)),
		)
	})
}

// Test that completions still work with an undownloaded module, golang/go#43333.
func TestUndownloadedModule(t *testing.T) {
	// mod.com depends on example.com, but only in a file that's hidden by a
	// build tag, so the IWL won't download example.com. That will cause errors
	// in the go list -m call performed by the imports package.
	const files = `
-- go.mod --
module mod.com

go 1.14

require example.com v1.2.3
-- go.sum --
example.com v1.2.3 h1:ihBTGWGjTU3V4ZJ9OmHITkU9WQ4lGdQkMjgyLFk0FaY=
example.com v1.2.3/go.mod h1:Y2Rc5rVWjWur0h3pd9aEvK5Pof8YKDANh9gHA2Maujo=
-- useblah.go --
// +build hidden

package pkg
import "example.com/blah"
var _ = blah.Name
-- mainmod/mainmod.go --
package mainmod

const Name = "mainmod"
`
	WithOptions(ProxyFiles(proxy)).Run(t, files, func(t *testing.T, env *Env) {
		env.CreateBuffer("import.go", "package pkg\nvar _ = mainmod.Name\n")
		env.SaveBuffer("import.go")
		content := env.ReadWorkspaceFile("import.go")
		if !strings.Contains(content, `import "mod.com/mainmod`) {
			t.Errorf("expected import of mod.com/mainmod in %q", content)
		}
	})
}

// Test that we can doctor the source code enough so the file is
// parseable and completion works as expected.
func TestSourceFixup(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.12
-- foo.go --
package foo

func _() {
	var s S
	if s.
}

type S struct {
	i int
}
`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("foo.go")
		completions := env.Completion(env.RegexpSearch("foo.go", `if s\.()`))
		diff := compareCompletionLabels([]string{"i"}, completions.Items)
		if diff != "" {
			t.Fatal(diff)
		}
	})
}

func TestCompletion_Issue45510(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

func _() {
	type a *a
	var aaaa1, aaaa2 a
	var _ a = aaaa

	type b a
	var bbbb1, bbbb2 b
	var _ b = bbbb
}

type (
	c *d
	d *e
	e **c
)

func _() {
	var (
		xxxxc c
		xxxxd d
		xxxxe e
	)

	var _ c = xxxx
	var _ d = xxxx
	var _ e = xxxx
}
`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")

		tests := []struct {
			re   string
			want []string
		}{
			{`var _ a = aaaa()`, []string{"aaaa1", "aaaa2"}},
			{`var _ b = bbbb()`, []string{"bbbb1", "bbbb2"}},
			{`var _ c = xxxx()`, []string{"xxxxc", "xxxxd", "xxxxe"}},
			{`var _ d = xxxx()`, []string{"xxxxc", "xxxxd", "xxxxe"}},
			{`var _ e = xxxx()`, []string{"xxxxc", "xxxxd", "xxxxe"}},
		}
		for _, tt := range tests {
			completions := env.Completion(env.RegexpSearch("main.go", tt.re))
			diff := compareCompletionLabels(tt.want, completions.Items)
			if diff != "" {
				t.Errorf("%s: %s", tt.re, diff)
			}
		}
	})
}

func TestCompletionDeprecation(t *testing.T) {
	const files = `
-- go.mod --
module test.com

go 1.16
-- prog.go --
package waste
// Deprecated, use newFoof
func fooFunc() bool {
	return false
}

// Deprecated
const badPi = 3.14

func doit() {
	if fooF
	panic()
	x := badP
}
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("prog.go")
		loc := env.RegexpSearch("prog.go", "if fooF")
		loc.Range.Start.Character += uint32(protocol.UTF16Len([]byte("if fooF")))
		completions := env.Completion(loc)
		diff := compareCompletionLabels([]string{"fooFunc"}, completions.Items)
		if diff != "" {
			t.Error(diff)
		}
		if completions.Items[0].Tags == nil {
			t.Errorf("expected Tags to show deprecation %#v", completions.Items[0].Tags)
		}
		loc = env.RegexpSearch("prog.go", "= badP")
		loc.Range.Start.Character += uint32(protocol.UTF16Len([]byte("= badP")))
		completions = env.Completion(loc)
		diff = compareCompletionLabels([]string{"badPi"}, completions.Items)
		if diff != "" {
			t.Error(diff)
		}
		if completions.Items[0].Tags == nil {
			t.Errorf("expected Tags to show deprecation %#v", completions.Items[0].Tags)
		}
	})
}

func TestUnimportedCompletion_VSCodeIssue1489(t *testing.T) {
	const src = `
-- go.mod --
module mod.com

go 1.14

-- main.go --
package main

import "fmt"

func main() {
	fmt.Println("a")
	math.Sqr
}
`
	WithOptions(
		WindowsLineEndings(),
	).Run(t, src, func(t *testing.T, env *Env) {
		// Trigger unimported completions for the mod.com package.
		env.OpenFile("main.go")
		env.Await(env.DoneWithOpen())
		loc := env.RegexpSearch("main.go", "Sqr()")
		completions := env.Completion(loc)
		if len(completions.Items) == 0 {
			t.Fatalf("no completion items")
		}
		env.AcceptCompletion(loc, completions.Items[0])
		env.Await(env.DoneWithChange())
		got := env.BufferText("main.go")
		want := "package main\r\n\r\nimport (\r\n\t\"fmt\"\r\n\t\"math\"\r\n)\r\n\r\nfunc main() {\r\n\tfmt.Println(\"a\")\r\n\tmath.Sqrt(${1:x float64})\r\n}\r\n"
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("unimported completion (-want +got):\n%s", diff)
		}
	})
}

func TestUnimportedCompletionHasPlaceholders60269(t *testing.T) {
	// We can't express this as a marker test because it doesn't support AcceptCompletion.
	const src = `
-- go.mod --
module example.com
go 1.12

-- a/a.go --
package a

var _ = b.F

-- b/b.go --
package b

func F0(a, b int, c float64) {}
func F1(int, chan *string) {}
`
	WithOptions(
		WindowsLineEndings(),
	).Run(t, src, func(t *testing.T, env *Env) {
		env.OpenFile("a/a.go")
		env.Await(env.DoneWithOpen())

		// The table lists the expected completions as they appear in Items.
		const common = "package a\r\n\r\nimport \"example.com/b\"\r\n\r\nvar _ = "
		for i, want := range []string{
			common + "b.F0(${1:a int}, ${2:b int}, ${3:c float64})\r\n",
			common + "b.F1(${1:_ int}, ${2:_ chan *string})\r\n",
		} {
			loc := env.RegexpSearch("a/a.go", "b.F()")
			completions := env.Completion(loc)
			if len(completions.Items) == 0 {
				t.Fatalf("no completion items")
			}
			saved := env.BufferText("a/a.go")
			env.AcceptCompletion(loc, completions.Items[i])
			env.Await(env.DoneWithChange())
			got := env.BufferText("a/a.go")
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("%d: unimported completion (-want +got):\n%s", i, diff)
			}
			env.SetBufferContent("a/a.go", saved) // restore
		}
	})
}

func TestPackageMemberCompletionAfterSyntaxError(t *testing.T) {
	// This test documents the current broken behavior due to golang/go#58833.
	const src = `
-- go.mod --
module mod.com

go 1.14

-- main.go --
package main

import "math"

func main() {
	math.Sqrt(,0)
	math.Ldex
}
`
	Run(t, src, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.Await(env.DoneWithOpen())
		loc := env.RegexpSearch("main.go", "Ldex()")
		completions := env.Completion(loc)
		if len(completions.Items) == 0 {
			t.Fatalf("no completion items")
		}
		env.AcceptCompletion(loc, completions.Items[0])
		env.Await(env.DoneWithChange())
		got := env.BufferText("main.go")
		// The completion of math.Ldex after the syntax error on the
		// previous line is not "math.Ldexp" but "math.Ldexmath.Abs".
		// (In VSCode, "Abs" wrongly appears in the completion menu.)
		// This is a consequence of poor error recovery in the parser
		// causing "math.Ldex" to become a BadExpr.
		want := "package main\n\nimport \"math\"\n\nfunc main() {\n\tmath.Sqrt(,0)\n\tmath.Ldexmath.Abs(${1:})\n}\n"
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("unimported completion (-want +got):\n%s", diff)
		}
	})
}

func TestCompleteAllFields(t *testing.T) {
	// This test verifies that completion results always include all struct fields.
	// See golang/go#53992.

	const src = `
-- go.mod --
module mod.com

go 1.18

-- p/p.go --
package p

import (
	"fmt"

	. "net/http"
	. "runtime"
	. "go/types"
	. "go/parser"
	. "go/ast"
)

type S struct {
	a, b, c, d, e, f, g, h, i, j, k, l, m int
	n, o, p, q, r, s, t, u, v, w, x, y, z int
}

func _() {
	var s S
	fmt.Println(s.)
}
`

	WithOptions(Settings{
		"completionBudget": "1ns", // must be non-zero as 0 => infinity
	}).Run(t, src, func(t *testing.T, env *Env) {
		wantFields := make(map[string]bool)
		for c := 'a'; c <= 'z'; c++ {
			wantFields[string(c)] = true
		}

		env.OpenFile("p/p.go")
		// Make an arbitrary edit to ensure we're not hitting the cache.
		env.EditBuffer("p/p.go", fake.NewEdit(0, 0, 0, 0, fmt.Sprintf("// current time: %v\n", time.Now())))
		loc := env.RegexpSearch("p/p.go", `s\.()`)
		completions := env.Completion(loc)
		gotFields := make(map[string]bool)
		for _, item := range completions.Items {
			if item.Kind == protocol.FieldCompletion {
				gotFields[item.Label] = true
			}
		}

		if diff := cmp.Diff(wantFields, gotFields); diff != "" {
			t.Errorf("Completion(...) returned mismatching fields (-want +got):\n%s", diff)
		}
	})
}

func TestDefinition(t *testing.T) {
	testenv.NeedsGo1Point(t, 17) // in go1.16, The FieldList in func x is not empty
	files := `
-- go.mod --
module mod.com

go 1.18
-- a_test.go --
package foo
`
	tests := []struct {
		line string   // the sole line in the buffer after the package statement
		pat  string   // the pattern to search for
		want []string // expected completions
	}{
		{"func T", "T", []string{"TestXxx(t *testing.T)", "TestMain(m *testing.M)"}},
		{"func T()", "T", []string{"TestMain", "Test"}},
		{"func TestM", "TestM", []string{"TestMain(m *testing.M)", "TestM(t *testing.T)"}},
		{"func TestM()", "TestM", []string{"TestMain"}},
		{"func TestMi", "TestMi", []string{"TestMi(t *testing.T)"}},
		{"func TestMi()", "TestMi", nil},
		{"func TestG", "TestG", []string{"TestG(t *testing.T)"}},
		{"func TestG(", "TestG", nil},
		{"func Ben", "B", []string{"BenchmarkXxx(b *testing.B)"}},
		{"func Ben(", "Ben", []string{"Benchmark"}},
		{"func BenchmarkFoo", "BenchmarkFoo", []string{"BenchmarkFoo(b *testing.B)"}},
		{"func BenchmarkFoo(", "BenchmarkFoo", nil},
		{"func Fuz", "F", []string{"FuzzXxx(f *testing.F)"}},
		{"func Fuz(", "Fuz", []string{"Fuzz"}},
		{"func Testx", "Testx", nil},
		{"func TestMe(t *testing.T)", "TestMe", nil},
		{"func Te(t *testing.T)", "Te", []string{"TestMain", "Test"}},
	}
	fname := "a_test.go"
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile(fname)
		env.Await(env.DoneWithOpen())
		for _, test := range tests {
			env.SetBufferContent(fname, "package foo\n"+test.line)
			loc := env.RegexpSearch(fname, test.pat)
			loc.Range.Start.Character += uint32(protocol.UTF16Len([]byte(test.pat)))
			completions := env.Completion(loc)
			if diff := compareCompletionLabels(test.want, completions.Items); diff != "" {
				t.Error(diff)
			}
		}
	})
}

// Test that completing a definition replaces source text when applied, golang/go#56852.
// Note: With go <= 1.16 the completions does not add parameters and fails these tests.
func TestDefinitionReplaceRange(t *testing.T) {
	testenv.NeedsGo1Point(t, 17)

	const mod = `
-- go.mod --
module mod.com

go 1.17
`

	tests := []struct {
		name          string
		before, after string
	}{
		{
			name: "func TestMa",
			before: `
package foo_test

func TestMa
`,
			after: `
package foo_test

func TestMain(m *testing.M)
`,
		},
		{
			name: "func TestSome",
			before: `
package foo_test

func TestSome
`,
			after: `
package foo_test

func TestSome(t *testing.T)
`,
		},
		{
			name: "func Bench",
			before: `
package foo_test

func Bench
`,
			// Note: Snippet with escaped }.
			after: `
package foo_test

func Benchmark${1:Xxx}(b *testing.B) {
	$0
\}
`,
		},
	}

	Run(t, mod, func(t *testing.T, env *Env) {
		env.CreateBuffer("foo_test.go", "")

		for _, tst := range tests {
			tst.before = strings.Trim(tst.before, "\n")
			tst.after = strings.Trim(tst.after, "\n")
			env.SetBufferContent("foo_test.go", tst.before)

			loc := env.RegexpSearch("foo_test.go", tst.name)
			loc.Range.Start.Character = uint32(protocol.UTF16Len([]byte(tst.name)))
			completions := env.Completion(loc)
			if len(completions.Items) == 0 {
				t.Fatalf("no completion items")
			}

			env.AcceptCompletion(loc, completions.Items[0])
			env.Await(env.DoneWithChange())
			if buf := env.BufferText("foo_test.go"); buf != tst.after {
				t.Errorf("%s:incorrect completion: got %q, want %q", tst.name, buf, tst.after)
			}
		}
	})
}

func TestGoWorkCompletion(t *testing.T) {
	const files = `
-- go.work --
go 1.18

use ./a
use ./a/ba
use ./a/b/
use ./dir/foo
use ./dir/foobar/
-- a/go.mod --
-- go.mod --
-- a/bar/go.mod --
-- a/b/c/d/e/f/go.mod --
-- dir/bar --
-- dir/foobar/go.mod --
`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("go.work")

		tests := []struct {
			re   string
			want []string
		}{
			{`use ()\.`, []string{".", "./a", "./a/bar", "./dir/foobar"}},
			{`use \.()`, []string{"", "/a", "/a/bar", "/dir/foobar"}},
			{`use \./()`, []string{"a", "a/bar", "dir/foobar"}},
			{`use ./a()`, []string{"", "/b/c/d/e/f", "/bar"}},
			{`use ./a/b()`, []string{"/c/d/e/f", "ar"}},
			{`use ./a/b/()`, []string{`c/d/e/f`}},
			{`use ./a/ba()`, []string{"r"}},
			{`use ./dir/foo()`, []string{"bar"}},
			{`use ./dir/foobar/()`, []string{}},
		}
		for _, tt := range tests {
			completions := env.Completion(env.RegexpSearch("go.work", tt.re))
			diff := compareCompletionLabels(tt.want, completions.Items)
			if diff != "" {
				t.Errorf("%s: %s", tt.re, diff)
			}
		}
	})
}
