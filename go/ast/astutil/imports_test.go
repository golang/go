// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package astutil

import (
	"bytes"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"reflect"
	"strconv"
	"testing"
)

var fset = token.NewFileSet()

func parse(t *testing.T, name, in string) *ast.File {
	file, err := parser.ParseFile(fset, name, in, parser.ParseComments)
	if err != nil {
		t.Fatalf("%s parse: %v", name, err)
	}
	return file
}

func print(t *testing.T, name string, f *ast.File) string {
	var buf bytes.Buffer
	if err := format.Node(&buf, fset, f); err != nil {
		t.Fatalf("%s gofmt: %v", name, err)
	}
	return string(buf.Bytes())
}

type test struct {
	name       string
	renamedPkg string
	pkg        string
	in         string
	out        string
	broken     bool // known broken
}

var addTests = []test{
	{
		name: "leave os alone",
		pkg:  "os",
		in: `package main

import (
	"os"
)
`,
		out: `package main

import (
	"os"
)
`,
	},
	{
		name: "import.1",
		pkg:  "os",
		in: `package main
`,
		out: `package main

import "os"
`,
	},
	{
		name: "import.2",
		pkg:  "os",
		in: `package main

// Comment
import "C"
`,
		out: `package main

// Comment
import "C"
import "os"
`,
	},
	{
		name: "import.3",
		pkg:  "os",
		in: `package main

// Comment
import "C"

import (
	"io"
	"utf8"
)
`,
		out: `package main

// Comment
import "C"

import (
	"io"
	"os"
	"utf8"
)
`,
	},
	{
		name: "import.17",
		pkg:  "x/y/z",
		in: `package main

// Comment
import "C"

import (
	"a"
	"b"

	"x/w"

	"d/f"
)
`,
		out: `package main

// Comment
import "C"

import (
	"a"
	"b"

	"x/w"
	"x/y/z"

	"d/f"
)
`,
	},
	{
		name: "import into singular group",
		pkg:  "bytes",
		in: `package main

import "os"

`,
		out: `package main

import (
	"bytes"
	"os"
)
`,
	},
	{
		name: "import into singular group with comment",
		pkg:  "bytes",
		in: `package main

import /* why */ /* comment here? */ "os"

`,
		out: `package main

import /* why */ /* comment here? */ (
	"bytes"
	"os"
)
`,
	},
	{
		name: "import into group with leading comment",
		pkg:  "strings",
		in: `package main

import (
	// comment before bytes
	"bytes"
	"os"
)

`,
		out: `package main

import (
	// comment before bytes
	"bytes"
	"os"
	"strings"
)
`,
	},
	{
		name:       "",
		renamedPkg: "fmtpkg",
		pkg:        "fmt",
		in: `package main

import "os"

`,
		out: `package main

import (
	fmtpkg "fmt"
	"os"
)
`,
	},
	{
		name: "struct comment",
		pkg:  "time",
		in: `package main

// This is a comment before a struct.
type T struct {
	t  time.Time
}
`,
		out: `package main

import "time"

// This is a comment before a struct.
type T struct {
	t time.Time
}
`,
	},
	{
		name: "issue 8729 import C",
		pkg:  "time",
		in: `package main

import "C"

// comment
type T time.Time
`,
		out: `package main

import "C"
import "time"

// comment
type T time.Time
`,
	},
	{
		name: "issue 8729 empty import",
		pkg:  "time",
		in: `package main

import ()

// comment
type T time.Time
`,
		out: `package main

import "time"

// comment
type T time.Time
`,
	},
	{
		name: "issue 8729 comment on package line",
		pkg:  "time",
		in: `package main // comment

type T time.Time
`,
		out: `package main // comment
import "time"

type T time.Time
`,
	},
	{
		name: "issue 8729 comment after package",
		pkg:  "time",
		in: `package main
// comment

type T time.Time
`,
		out: `package main

import "time"

// comment

type T time.Time
`,
	},
	{
		name: "issue 8729 comment before and on package line",
		pkg:  "time",
		in: `// comment before
package main // comment on

type T time.Time
`,
		out: `// comment before
package main // comment on
import "time"

type T time.Time
`,
	},

	// Issue 9961: Match prefixes using path segments rather than bytes
	{
		name: "issue 9961",
		pkg:  "regexp",
		in: `package main

import (
	"flag"
	"testing"

	"rsc.io/p"
)
`,
		out: `package main

import (
	"flag"
	"regexp"
	"testing"

	"rsc.io/p"
)
`,
	},
}

func TestAddImport(t *testing.T) {
	for _, test := range addTests {
		file := parse(t, test.name, test.in)
		var before bytes.Buffer
		ast.Fprint(&before, fset, file, nil)
		AddNamedImport(fset, file, test.renamedPkg, test.pkg)
		if got := print(t, test.name, file); got != test.out {
			if test.broken {
				t.Logf("%s is known broken:\ngot: %s\nwant: %s", test.name, got, test.out)
			} else {
				t.Errorf("%s:\ngot: %s\nwant: %s", test.name, got, test.out)
			}
			var after bytes.Buffer
			ast.Fprint(&after, fset, file, nil)

			t.Logf("AST before:\n%s\nAST after:\n%s\n", before.String(), after.String())
		}
	}
}

func TestDoubleAddImport(t *testing.T) {
	file := parse(t, "doubleimport", "package main\n")
	AddImport(fset, file, "os")
	AddImport(fset, file, "bytes")
	want := `package main

import (
	"bytes"
	"os"
)
`
	if got := print(t, "doubleimport", file); got != want {
		t.Errorf("got: %s\nwant: %s", got, want)
	}
}

func TestDoubleAddNamedImport(t *testing.T) {
	file := parse(t, "doublenamedimport", "package main\n")
	AddNamedImport(fset, file, "o", "os")
	AddNamedImport(fset, file, "i", "io")
	want := `package main

import (
	i "io"
	o "os"
)
`
	if got := print(t, "doublenamedimport", file); got != want {
		t.Errorf("got: %s\nwant: %s", got, want)
	}
}

// Part of issue 8729.
func TestDoubleAddImportWithDeclComment(t *testing.T) {
	file := parse(t, "doubleimport", `package main

import (
)

// comment
type I int
`)
	// The AddImport order here matters.
	AddImport(fset, file, "golang.org/x/tools/go/ast/astutil")
	AddImport(fset, file, "os")
	want := `package main

import (
	"golang.org/x/tools/go/ast/astutil"
	"os"
)

// comment
type I int
`
	if got := print(t, "doubleimport_with_decl_comment", file); got != want {
		t.Errorf("got: %s\nwant: %s", got, want)
	}
}

var deleteTests = []test{
	{
		name: "import.4",
		pkg:  "os",
		in: `package main

import (
	"os"
)
`,
		out: `package main
`,
	},
	{
		name: "import.5",
		pkg:  "os",
		in: `package main

// Comment
import "C"
import "os"
`,
		out: `package main

// Comment
import "C"
`,
	},
	{
		name: "import.6",
		pkg:  "os",
		in: `package main

// Comment
import "C"

import (
	"io"
	"os"
	"utf8"
)
`,
		out: `package main

// Comment
import "C"

import (
	"io"
	"utf8"
)
`,
	},
	{
		name: "import.7",
		pkg:  "io",
		in: `package main

import (
	"io"   // a
	"os"   // b
	"utf8" // c
)
`,
		out: `package main

import (
	// a
	"os"   // b
	"utf8" // c
)
`,
	},
	{
		name: "import.8",
		pkg:  "os",
		in: `package main

import (
	"io"   // a
	"os"   // b
	"utf8" // c
)
`,
		out: `package main

import (
	"io" // a
	// b
	"utf8" // c
)
`,
	},
	{
		name: "import.9",
		pkg:  "utf8",
		in: `package main

import (
	"io"   // a
	"os"   // b
	"utf8" // c
)
`,
		out: `package main

import (
	"io" // a
	"os" // b
	// c
)
`,
	},
	{
		name: "import.10",
		pkg:  "io",
		in: `package main

import (
	"io"
	"os"
	"utf8"
)
`,
		out: `package main

import (
	"os"
	"utf8"
)
`,
	},
	{
		name: "import.11",
		pkg:  "os",
		in: `package main

import (
	"io"
	"os"
	"utf8"
)
`,
		out: `package main

import (
	"io"
	"utf8"
)
`,
	},
	{
		name: "import.12",
		pkg:  "utf8",
		in: `package main

import (
	"io"
	"os"
	"utf8"
)
`,
		out: `package main

import (
	"io"
	"os"
)
`,
	},
	{
		name: "handle.raw.quote.imports",
		pkg:  "os",
		in:   "package main\n\nimport `os`",
		out: `package main
`,
	},
	{
		name: "import.13",
		pkg:  "io",
		in: `package main

import (
	"fmt"

	"io"
	"os"
	"utf8"

	"go/format"
)
`,
		out: `package main

import (
	"fmt"

	"os"
	"utf8"

	"go/format"
)
`,
	},
	{
		name: "import.14",
		pkg:  "io",
		in: `package main

import (
	"fmt" // a

	"io"   // b
	"os"   // c
	"utf8" // d

	"go/format" // e
)
`,
		out: `package main

import (
	"fmt" // a

	// b
	"os"   // c
	"utf8" // d

	"go/format" // e
)
`,
	},
	{
		name: "import.15",
		pkg:  "double",
		in: `package main

import (
	"double"
	"double"
)
`,
		out: `package main
`,
	},
	{
		name: "import.16",
		pkg:  "bubble",
		in: `package main

import (
	"toil"
	"bubble"
	"bubble"
	"trouble"
)
`,
		out: `package main

import (
	"toil"
	"trouble"
)
`,
	},
	{
		name: "import.17",
		pkg:  "quad",
		in: `package main

import (
	"quad"
	"quad"
)

import (
	"quad"
	"quad"
)
`,
		out: `package main
`,
	},
}

func TestDeleteImport(t *testing.T) {
	for _, test := range deleteTests {
		file := parse(t, test.name, test.in)
		DeleteImport(fset, file, test.pkg)
		if got := print(t, test.name, file); got != test.out {
			t.Errorf("%s:\ngot: %s\nwant: %s", test.name, got, test.out)
		}
	}
}

type rewriteTest struct {
	name   string
	srcPkg string
	dstPkg string
	in     string
	out    string
}

var rewriteTests = []rewriteTest{
	{
		name:   "import.13",
		srcPkg: "utf8",
		dstPkg: "encoding/utf8",
		in: `package main

import (
	"io"
	"os"
	"utf8" // thanks ken
)
`,
		out: `package main

import (
	"encoding/utf8" // thanks ken
	"io"
	"os"
)
`,
	},
	{
		name:   "import.14",
		srcPkg: "asn1",
		dstPkg: "encoding/asn1",
		in: `package main

import (
	"asn1"
	"crypto"
	"crypto/rsa"
	_ "crypto/sha1"
	"crypto/x509"
	"crypto/x509/pkix"
	"time"
)

var x = 1
`,
		out: `package main

import (
	"crypto"
	"crypto/rsa"
	_ "crypto/sha1"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"time"
)

var x = 1
`,
	},
	{
		name:   "import.15",
		srcPkg: "url",
		dstPkg: "net/url",
		in: `package main

import (
	"bufio"
	"net"
	"path"
	"url"
)

var x = 1 // comment on x, not on url
`,
		out: `package main

import (
	"bufio"
	"net"
	"net/url"
	"path"
)

var x = 1 // comment on x, not on url
`,
	},
	{
		name:   "import.16",
		srcPkg: "http",
		dstPkg: "net/http",
		in: `package main

import (
	"flag"
	"http"
	"log"
	"text/template"
)

var addr = flag.String("addr", ":1718", "http service address") // Q=17, R=18
`,
		out: `package main

import (
	"flag"
	"log"
	"net/http"
	"text/template"
)

var addr = flag.String("addr", ":1718", "http service address") // Q=17, R=18
`,
	},
}

func TestRewriteImport(t *testing.T) {
	for _, test := range rewriteTests {
		file := parse(t, test.name, test.in)
		RewriteImport(fset, file, test.srcPkg, test.dstPkg)
		if got := print(t, test.name, file); got != test.out {
			t.Errorf("%s:\ngot: %s\nwant: %s", test.name, got, test.out)
		}
	}
}

var importsTests = []struct {
	name string
	in   string
	want [][]string
}{
	{
		name: "no packages",
		in: `package foo
`,
		want: nil,
	},
	{
		name: "one group",
		in: `package foo

import (
	"fmt"
	"testing"
)
`,
		want: [][]string{{"fmt", "testing"}},
	},
	{
		name: "four groups",
		in: `package foo

import "C"
import (
	"fmt"
	"testing"

	"appengine"

	"myproject/mylib1"
	"myproject/mylib2"
)
`,
		want: [][]string{
			{"C"},
			{"fmt", "testing"},
			{"appengine"},
			{"myproject/mylib1", "myproject/mylib2"},
		},
	},
	{
		name: "multiple factored groups",
		in: `package foo

import (
	"fmt"
	"testing"

	"appengine"
)
import (
	"reflect"

	"bytes"
)
`,
		want: [][]string{
			{"fmt", "testing"},
			{"appengine"},
			{"reflect"},
			{"bytes"},
		},
	},
}

func unquote(s string) string {
	res, err := strconv.Unquote(s)
	if err != nil {
		return "could_not_unquote"
	}
	return res
}

func TestImports(t *testing.T) {
	fset := token.NewFileSet()
	for _, test := range importsTests {
		f, err := parser.ParseFile(fset, "test.go", test.in, 0)
		if err != nil {
			t.Errorf("%s: %v", test.name, err)
			continue
		}
		var got [][]string
		for _, group := range Imports(fset, f) {
			var b []string
			for _, spec := range group {
				b = append(b, unquote(spec.Path.Value))
			}
			got = append(got, b)
		}
		if !reflect.DeepEqual(got, test.want) {
			t.Errorf("Imports(%s)=%v, want %v", test.name, got, test.want)
		}
	}
}
