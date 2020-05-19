package source

import (
	"testing"
)

type data struct {
	input, want string
}

func TestImportPrefix(t *testing.T) {
	var tdata = []data{
		{"package foo\n", "package foo\n"},
		{"package foo\n\nfunc f(){}\n", "package foo\n"},
		{"package foo\n\nimport \"fmt\"\n", "package foo\n\nimport \"fmt\""},
		{"package foo\nimport (\n\"fmt\"\n)\n", "package foo\nimport (\n\"fmt\"\n)"},
		{"\n\n\npackage foo\n", "\n\n\npackage foo\n"},
		{"// hi \n\npackage foo //xx\nfunc _(){}\n", "// hi \n\npackage foo //xx\n"},
		{"package foo //hi\n", "package foo //hi\n"},
		{"//hi\npackage foo\n//a\n\n//b\n", "//hi\npackage foo\n//a\n\n//b\n"},
		{"package a\n\nimport (\n  \"fmt\"\n)\n//hi\n",
			"package a\n\nimport (\n  \"fmt\"\n)\n//hi\n"},
	}
	for i, x := range tdata {
		got := importPrefix([]byte(x.input))
		if got != x.want {
			t.Errorf("%d: got\n%q, wanted\n%q", i, got, x.want)
		}
	}
}
