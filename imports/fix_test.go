// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"flag"
	"go/build"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

var only = flag.String("only", "", "If non-empty, the fix test to run")

var tests = []struct {
	name    string
	in, out string
}{
	// Adding an import to an existing parenthesized import
	{
		name: "factored_imports_add",
		in: `package foo
import (
  "fmt"
)
func bar() {
var b bytes.Buffer
fmt.Println(b.String())
}
`,
		out: `package foo

import (
	"bytes"
	"fmt"
)

func bar() {
	var b bytes.Buffer
	fmt.Println(b.String())
}
`,
	},

	// Adding an import to an existing parenthesized import,
	// verifying it goes into the first section.
	{
		name: "factored_imports_add_first_sec",
		in: `package foo
import (
  "fmt"

  "appengine"
)
func bar() {
var b bytes.Buffer
_ = appengine.IsDevServer
fmt.Println(b.String())
}
`,
		out: `package foo

import (
	"bytes"
	"fmt"

	"appengine"
)

func bar() {
	var b bytes.Buffer
	_ = appengine.IsDevServer
	fmt.Println(b.String())
}
`,
	},

	// Adding an import to an existing parenthesized import,
	// verifying it goes into the first section. (test 2)
	{
		name: "factored_imports_add_first_sec_2",
		in: `package foo
import (
  "fmt"

  "appengine"
)
func bar() {
_ = math.NaN
_ = fmt.Sprintf
_ = appengine.IsDevServer
}
`,
		out: `package foo

import (
	"fmt"
	"math"

	"appengine"
)

func bar() {
	_ = math.NaN
	_ = fmt.Sprintf
	_ = appengine.IsDevServer
}
`,
	},

	// Adding a new import line, without parens
	{
		name: "add_import_section",
		in: `package foo
func bar() {
var b bytes.Buffer
}
`,
		out: `package foo

import "bytes"

func bar() {
	var b bytes.Buffer
}
`,
	},

	// Adding two new imports, which should make a parenthesized import decl.
	{
		name: "add_import_paren_section",
		in: `package foo
func bar() {
_, _ := bytes.Buffer, zip.NewReader
}
`,
		out: `package foo

import (
	"archive/zip"
	"bytes"
)

func bar() {
	_, _ := bytes.Buffer, zip.NewReader
}
`,
	},

	// Make sure we don't add things twice
	{
		name: "no_double_add",
		in: `package foo
func bar() {
_, _ := bytes.Buffer, bytes.NewReader
}
`,
		out: `package foo

import "bytes"

func bar() {
	_, _ := bytes.Buffer, bytes.NewReader
}
`,
	},

	// Remove unused imports, 1 of a factored block
	{
		name: "remove_unused_1_of_2",
		in: `package foo
import (
"bytes"
"fmt"
)

func bar() {
_, _ := bytes.Buffer, bytes.NewReader
}
`,
		out: `package foo

import "bytes"

func bar() {
	_, _ := bytes.Buffer, bytes.NewReader
}
`,
	},

	// Remove unused imports, 2 of 2
	{
		name: "remove_unused_2_of_2",
		in: `package foo
import (
"bytes"
"fmt"
)

func bar() {
}
`,
		out: `package foo

func bar() {
}
`,
	},

	// Remove unused imports, 1 of 1
	{
		name: "remove_unused_1_of_1",
		in: `package foo

import "fmt"

func bar() {
}
`,
		out: `package foo

func bar() {
}
`,
	},

	// Don't remove empty imports.
	{
		name: "dont_remove_empty_imports",
		in: `package foo
import (
_ "image/png"
_ "image/jpeg"
)
`,
		out: `package foo

import (
	_ "image/jpeg"
	_ "image/png"
)
`,
	},

	// Don't remove dot imports.
	{
		name: "dont_remove_dot_imports",
		in: `package foo
import (
. "foo"
. "bar"
)
`,
		out: `package foo

import (
	. "bar"
	. "foo"
)
`,
	},

	// Skip refs the parser can resolve.
	{
		name: "skip_resolved_refs",
		in: `package foo

func f() {
	type t struct{ Println func(string) }
	fmt := t{Println: func(string) {}}
	fmt.Println("foo")
}
`,
		out: `package foo

func f() {
	type t struct{ Println func(string) }
	fmt := t{Println: func(string) {}}
	fmt.Println("foo")
}
`,
	},

	// Do not add a package we already have a resolution for.
	{
		name: "skip_template",
		in: `package foo

import "html/template"

func f() { t = template.New("sometemplate") }
`,
		out: `package foo

import "html/template"

func f() { t = template.New("sometemplate") }
`,
	},

	// Don't touch cgo
	{
		name: "cgo",
		in: `package foo

/*
#include <foo.h>
*/
import "C"
`,
		out: `package foo

/*
#include <foo.h>
*/
import "C"
`,
	},

	// Put some things in their own section
	{
		name: "make_sections",
		in: `package foo

import (
"os"
)

func foo () {
_, _ = os.Args, fmt.Println
_, _ = appengine.FooSomething, user.Current
}
`,
		out: `package foo

import (
	"fmt"
	"os"

	"appengine"
	"appengine/user"
)

func foo() {
	_, _ = os.Args, fmt.Println
	_, _ = appengine.FooSomething, user.Current
}
`,
	},

	// Delete existing empty import block
	{
		name: "delete_empty_import_block",
		in: `package foo

import ()
`,
		out: `package foo
`,
	},

	// Use existing empty import block
	{
		name: "use_empty_import_block",
		in: `package foo

import ()

func f() {
	_ = fmt.Println
}
`,
		out: `package foo

import "fmt"

func f() {
	_ = fmt.Println
}
`,
	},

	// Blank line before adding new section.
	{
		name: "blank_line_before_new_group",
		in: `package foo

import (
	"fmt"
	"net"
)

func f() {
	_ = net.Dial
	_ = fmt.Printf
	_ = snappy.Foo
}
`,
		out: `package foo

import (
	"fmt"
	"net"

	"code.google.com/p/snappy-go/snappy"
)

func f() {
	_ = net.Dial
	_ = fmt.Printf
	_ = snappy.Foo
}
`,
	},

	// Blank line between standard library and third-party stuff.
	{
		name: "blank_line_separating_std_and_third_party",
		in: `package foo

import (
	"code.google.com/p/snappy-go/snappy"
	"fmt"
	"net"
)

func f() {
	_ = net.Dial
	_ = fmt.Printf
	_ = snappy.Foo
}
`,
		out: `package foo

import (
	"fmt"
	"net"

	"code.google.com/p/snappy-go/snappy"
)

func f() {
	_ = net.Dial
	_ = fmt.Printf
	_ = snappy.Foo
}
`,
	},

	// golang.org/issue/6884
	{
		name: "issue 6884",
		in: `package main

// A comment
func main() {
	fmt.Println("Hello, world")
}
`,
		out: `package main

import "fmt"

// A comment
func main() {
	fmt.Println("Hello, world")
}
`,
	},
}

func TestFixImports(t *testing.T) {
	simplePkgs := map[string]string{
		"fmt":       "fmt",
		"os":        "os",
		"math":      "math",
		"appengine": "appengine",
		"user":      "appengine/user",
		"zip":       "archive/zip",
		"bytes":     "bytes",
		"snappy":    "code.google.com/p/snappy-go/snappy",
	}
	findImport = func(pkgName string, symbols map[string]bool) (string, error) {
		return simplePkgs[pkgName], nil
	}

	for _, tt := range tests {
		if *only != "" && tt.name != *only {
			continue
		}
		buf, err := Process("foo.go", []byte(tt.in), nil)
		if err != nil {
			t.Errorf("error on %q: %v", tt.name, err)
			continue
		}
		if got := string(buf); got != tt.out {
			t.Errorf("results diff on %q\nGOT:\n%s\nWANT:\n%s\n", tt.name, got, tt.out)
		}
	}
}

func TestFindImportGoPath(t *testing.T) {
	goroot, err := ioutil.TempDir("", "goimports-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(goroot)
	// Test against imaginary bits/bytes package in std lib
	bytesDir := filepath.Join(goroot, "src", "pkg", "bits", "bytes")
	if err := os.MkdirAll(bytesDir, 0755); err != nil {
		t.Fatal(err)
	}
	bytesSrcPath := filepath.Join(bytesDir, "bytes.go")
	bytesPkgPath := "bits/bytes"
	bytesSrc := []byte(`package bytes

type Buffer2 struct {}
`)
	if err := ioutil.WriteFile(bytesSrcPath, bytesSrc, 0775); err != nil {
		t.Fatal(err)
	}
	oldGOROOT := build.Default.GOROOT
	oldGOPATH := build.Default.GOPATH
	build.Default.GOROOT = goroot
	build.Default.GOPATH = ""
	defer func() {
		build.Default.GOROOT = oldGOROOT
		build.Default.GOPATH = oldGOPATH
	}()

	got, err := findImportGoPath("bytes", map[string]bool{"Buffer2": true})
	if err != nil {
		t.Fatal(err)
	}
	if got != bytesPkgPath {
		t.Errorf(`findImportGoPath("bytes", Buffer2 ...)=%q, want "%s"`, got, bytesPkgPath)
	}

	got, err = findImportGoPath("bytes", map[string]bool{"Missing": true})
	if err != nil {
		t.Fatal(err)
	}
	if got != "" {
		t.Errorf(`findImportGoPath("bytes", Missing ...)=%q, want ""`, got)
	}
}
