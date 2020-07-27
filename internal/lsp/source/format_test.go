package source

import (
	"fmt"
	"testing"

	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/diff/myers"
)

func TestImportPrefix(t *testing.T) {
	for i, tt := range []struct {
		input, want string
	}{
		{"package foo", "package foo"},
		{"package foo\n", "package foo\n"},
		{"package foo\n\nfunc f(){}\n", "package foo\n"},
		{"package foo\n\nimport \"fmt\"\n", "package foo\n\nimport \"fmt\""},
		{"package foo\nimport (\n\"fmt\"\n)\n", "package foo\nimport (\n\"fmt\"\n)"},
		{"\n\n\npackage foo\n", "\n\n\npackage foo\n"},
		{"// hi \n\npackage foo //xx\nfunc _(){}\n", "// hi \n\npackage foo //xx\n"},
		{"package foo //hi\n", "package foo //hi\n"},
		{"//hi\npackage foo\n//a\n\n//b\n", "//hi\npackage foo\n//a\n\n//b\n"},
		{
			"package a\n\nimport (\n  \"fmt\"\n)\n//hi\n",
			"package a\n\nimport (\n  \"fmt\"\n)\n//hi\n",
		},
		{`package a /*hi*/`, `package a /*hi*/`},
		{"package main\r\n\r\nimport \"go/types\"\r\n\r\n/*\r\n\r\n */\r\n", "package main\r\n\r\nimport \"go/types\"\r\n\r\n/*\r\n\r\n */\r\n"},
		{"package x; import \"os\"; func f() {}\n\n", "package x; import \"os\""},
		{"package x; func f() {fmt.Println()}\n\n", "package x"},
	} {
		got := importPrefix([]byte(tt.input))
		if got != tt.want {
			t.Errorf("%d: failed for %q:\n%s", i, tt.input, diffStr(tt.want, got))
		}
	}
}

func diffStr(want, got string) string {
	if want == got {
		return ""
	}
	// Add newlines to avoid newline messages in diff.
	want += "\n"
	got += "\n"
	d := myers.ComputeEdits("", want, got)
	return fmt.Sprintf("%q", diff.ToUnified("want", "got", want, d))
}
