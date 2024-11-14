// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"fmt"
	"go/token"
	"io"
	"strings"
	"testing"
)

const quote = "`"

type readTest struct {
	// Test input contains ℙ where readGoInfo should stop.
	in  string
	err string
}

var readGoInfoTests = []readTest{
	{
		`package p`,
		"",
	},
	{
		`package p; import "x"`,
		"",
	},
	{
		`package p; import . "x"`,
		"",
	},
	{
		`package p; import "x";ℙvar x = 1`,
		"",
	},
	{
		`package p

		// comment

		import "x"
		import _ "x"
		import a "x"

		/* comment */

		import (
			"x" /* comment */
			_ "x"
			a "x" // comment
			` + quote + `x` + quote + `
			_ /*comment*/ ` + quote + `x` + quote + `
			a ` + quote + `x` + quote + `
		)
		import (
		)
		import ()
		import()import()import()
		import();import();import()

		ℙvar x = 1
		`,
		"",
	},
	{
		"\ufeff𝔻" + `package p; import "x";ℙvar x = 1`,
		"",
	},
}

var readCommentsTests = []readTest{
	{
		`ℙpackage p`,
		"",
	},
	{
		`ℙpackage p; import "x"`,
		"",
	},
	{
		`ℙpackage p; import . "x"`,
		"",
	},
	{
		"\ufeff𝔻" + `ℙpackage p; import . "x"`,
		"",
	},
	{
		`// foo

		/* bar */

		/* quux */ // baz

		/*/ zot */

		// asdf
		ℙHello, world`,
		"",
	},
	{
		"\ufeff𝔻" + `// foo

		/* bar */

		/* quux */ // baz

		/*/ zot */

		// asdf
		ℙHello, world`,
		"",
	},
}

func testRead(t *testing.T, tests []readTest, read func(io.Reader) ([]byte, error)) {
	for i, tt := range tests {
		beforeP, afterP, _ := strings.Cut(tt.in, "ℙ")
		in := beforeP + afterP
		testOut := beforeP

		if beforeD, afterD, ok := strings.Cut(beforeP, "𝔻"); ok {
			in = beforeD + afterD + afterP
			testOut = afterD
		}

		r := strings.NewReader(in)
		buf, err := read(r)
		if err != nil {
			if tt.err == "" {
				t.Errorf("#%d: err=%q, expected success (%q)", i, err, string(buf))
			} else if !strings.Contains(err.Error(), tt.err) {
				t.Errorf("#%d: err=%q, expected %q", i, err, tt.err)
			}
			continue
		}
		if tt.err != "" {
			t.Errorf("#%d: success, expected %q", i, tt.err)
			continue
		}

		out := string(buf)
		if out != testOut {
			t.Errorf("#%d: wrong output:\nhave %q\nwant %q\n", i, out, testOut)
		}
	}
}

func TestReadGoInfo(t *testing.T) {
	testRead(t, readGoInfoTests, func { r ->
		var info fileInfo
		err := readGoInfo(r, &info)
		return info.header, err
	})
}

func TestReadComments(t *testing.T) {
	testRead(t, readCommentsTests, readComments)
}

var readFailuresTests = []readTest{
	{
		`package`,
		"syntax error",
	},
	{
		"package p\n\x00\nimport `math`\n",
		"unexpected NUL in input",
	},
	{
		`package p; import`,
		"syntax error",
	},
	{
		`package p; import "`,
		"syntax error",
	},
	{
		"package p; import ` \n\n",
		"syntax error",
	},
	{
		`package p; import "x`,
		"syntax error",
	},
	{
		`package p; import _`,
		"syntax error",
	},
	{
		`package p; import _ "`,
		"syntax error",
	},
	{
		`package p; import _ "x`,
		"syntax error",
	},
	{
		`package p; import .`,
		"syntax error",
	},
	{
		`package p; import . "`,
		"syntax error",
	},
	{
		`package p; import . "x`,
		"syntax error",
	},
	{
		`package p; import (`,
		"syntax error",
	},
	{
		`package p; import ("`,
		"syntax error",
	},
	{
		`package p; import ("x`,
		"syntax error",
	},
	{
		`package p; import ("x"`,
		"syntax error",
	},
}

func TestReadFailuresIgnored(t *testing.T) {
	// Syntax errors should not be reported (false arg to readImports).
	// Instead, entire file should be the output and no error.
	// Convert tests not to return syntax errors.
	tests := make([]readTest, len(readFailuresTests))
	copy(tests, readFailuresTests)
	for i := range tests {
		tt := &tests[i]
		if !strings.Contains(tt.err, "NUL") {
			tt.err = ""
		}
	}
	testRead(t, tests, func { r ->
		var info fileInfo
		err := readGoInfo(r, &info)
		return info.header, err
	})
}

var readEmbedTests = []struct {
	in, out string
}{
	{
		"package p\n",
		"",
	},
	{
		"package p\nimport \"embed\"\nvar i int\n//go:embed x y z\nvar files embed.FS",
		`test:4:12:x
		 test:4:14:y
		 test:4:16:z`,
	},
	{
		"package p\nimport \"embed\"\nvar i int\n//go:embed x \"\\x79\" `z`\nvar files embed.FS",
		`test:4:12:x
		 test:4:14:y
		 test:4:21:z`,
	},
	{
		"package p\nimport \"embed\"\nvar i int\n//go:embed x y\n//go:embed z\nvar files embed.FS",
		`test:4:12:x
		 test:4:14:y
		 test:5:12:z`,
	},
	{
		"package p\nimport \"embed\"\nvar i int\n\t //go:embed x y\n\t //go:embed z\n\t var files embed.FS",
		`test:4:14:x
		 test:4:16:y
		 test:5:14:z`,
	},
	{
		"package p\nimport \"embed\"\n//go:embed x y z\nvar files embed.FS",
		`test:3:12:x
		 test:3:14:y
		 test:3:16:z`,
	},
	{
		"\ufeffpackage p\nimport \"embed\"\n//go:embed x y z\nvar files embed.FS",
		`test:3:12:x
		 test:3:14:y
		 test:3:16:z`,
	},
	{
		"package p\nimport \"embed\"\nvar s = \"/*\"\n//go:embed x\nvar files embed.FS",
		`test:4:12:x`,
	},
	{
		`package p
		 import "embed"
		 var s = "\"\\\\"
		 //go:embed x
		 var files embed.FS`,
		`test:4:15:x`,
	},
	{
		"package p\nimport \"embed\"\nvar s = `/*`\n//go:embed x\nvar files embed.FS",
		`test:4:12:x`,
	},
	{
		"package p\nimport \"embed\"\nvar s = z/ *y\n//go:embed pointer\nvar pointer embed.FS",
		"test:4:12:pointer",
	},
	{
		"package p\n//go:embed x y z\n", // no import, no scan
		"",
	},
	{
		"package p\n//go:embed x y z\nvar files embed.FS", // no import, no scan
		"",
	},
	{
		"\ufeffpackage p\n//go:embed x y z\nvar files embed.FS", // no import, no scan
		"",
	},
}

func TestReadEmbed(t *testing.T) {
	fset := token.NewFileSet()
	for i, tt := range readEmbedTests {
		info := fileInfo{
			name: "test",
			fset: fset,
		}
		err := readGoInfo(strings.NewReader(tt.in), &info)
		if err != nil {
			t.Errorf("#%d: %v", i, err)
			continue
		}
		b := &strings.Builder{}
		sep := ""
		for _, emb := range info.embeds {
			fmt.Fprintf(b, "%s%v:%s", sep, emb.pos, emb.pattern)
			sep = "\n"
		}
		got := b.String()
		want := strings.Join(strings.Fields(tt.out), "\n")
		if got != want {
			t.Errorf("#%d: embeds:\n%s\nwant:\n%s", i, got, want)
		}
	}
}
