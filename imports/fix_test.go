// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"bytes"
	"flag"
	"go/build"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"sync"
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

	// golang.org/issue/7132
	{
		name: "issue 7132",
		in: `package main

import (
"fmt"

"gu"
"github.com/foo/bar"
)

var (
a = bar.a
b = gu.a
c = fmt.Printf
)
`,
		out: `package main

import (
	"fmt"

	"gu"

	"github.com/foo/bar"
)

var (
	a = bar.a
	b = gu.a
	c = fmt.Printf
)
`,
	},

	{
		name: "renamed package",
		in: `package main

var _ = str.HasPrefix
`,
		out: `package main

import str "strings"

var _ = str.HasPrefix
`,
	},

	{
		name: "fragment with main",
		in:   `func main(){fmt.Println("Hello, world")}`,
		out: `package main

import "fmt"

func main() { fmt.Println("Hello, world") }
`,
	},

	{
		name: "fragment without main",
		in:   `func notmain(){fmt.Println("Hello, world")}`,
		out: `import "fmt"

func notmain() { fmt.Println("Hello, world") }`,
	},

	// Remove first import within in a 2nd/3rd/4th/etc. section.
	// golang.org/issue/7679
	{
		name: "issue 7679",
		in: `package main

import (
	"fmt"

	"github.com/foo/bar"
	"github.com/foo/qux"
)

func main() {
	var _ = fmt.Println
	//var _ = bar.A
	var _ = qux.B
}
`,
		out: `package main

import (
	"fmt"

	"github.com/foo/qux"
)

func main() {
	var _ = fmt.Println
	//var _ = bar.A
	var _ = qux.B
}
`,
	},

	// Blank line can be added before all types of import declarations.
	// golang.org/issue/7866
	{
		name: "issue 7866",
		in: `package main

import (
	"fmt"
	renamed_bar "github.com/foo/bar"

	. "github.com/foo/baz"
	"io"

	_ "github.com/foo/qux"
	"strings"
)

func main() {
	_, _, _, _, _ = fmt.Errorf, io.Copy, strings.Contains, renamed_bar.A, B
}
`,
		out: `package main

import (
	"fmt"

	renamed_bar "github.com/foo/bar"

	"io"

	. "github.com/foo/baz"

	"strings"

	_ "github.com/foo/qux"
)

func main() {
	_, _, _, _, _ = fmt.Errorf, io.Copy, strings.Contains, renamed_bar.A, B
}
`,
	},

	// Non-idempotent comment formatting
	// golang.org/issue/8035
	{
		name: "issue 8035",
		in: `package main

import (
	"fmt"                     // A
	"go/ast"                  // B
	_ "launchpad.net/gocheck" // C
)

func main() { _, _ = fmt.Print, ast.Walk }
`,
		out: `package main

import (
	"fmt"    // A
	"go/ast" // B

	_ "launchpad.net/gocheck" // C
)

func main() { _, _ = fmt.Print, ast.Walk }
`,
	},

	// Failure to delete all duplicate imports
	// golang.org/issue/8459
	{
		name: "issue 8459",
		in: `package main

import (
	"fmt"
	"log"
	"log"
	"math"
)

func main() { fmt.Println("pi:", math.Pi) }
`,
		out: `package main

import (
	"fmt"
	"math"
)

func main() { fmt.Println("pi:", math.Pi) }
`,
	},

	// Too aggressive prefix matching
	// golang.org/issue/9961
	{
		name: "issue 9961",
		in: `package p

import (
	"zip"

	"rsc.io/p"
)

var (
	_ = fmt.Print
	_ = zip.Store
	_ p.P
	_ = regexp.Compile
)
`,
		out: `package p

import (
	"fmt"
	"regexp"
	"zip"

	"rsc.io/p"
)

var (
	_ = fmt.Print
	_ = zip.Store
	_ p.P
	_ = regexp.Compile
)
`,
	},

	// Unused named import is mistaken for unnamed import
	// golang.org/issue/8149
	{
		name: "issue 8149",
		in: `package main

import foo "fmt"

func main() { fmt.Println() }
`,
		out: `package main

import "fmt"

func main() { fmt.Println() }
`,
	},
}

func TestFixImports(t *testing.T) {
	simplePkgs := map[string]string{
		"appengine": "appengine",
		"bytes":     "bytes",
		"fmt":       "fmt",
		"math":      "math",
		"os":        "os",
		"p":         "rsc.io/p",
		"regexp":    "regexp",
		"snappy":    "code.google.com/p/snappy-go/snappy",
		"str":       "strings",
		"user":      "appengine/user",
		"zip":       "archive/zip",
	}
	old := findImport
	defer func() {
		findImport = old
	}()
	findImport = func(pkgName string, symbols map[string]bool, filename string) (string, bool, error) {
		return simplePkgs[pkgName], pkgName == "str", nil
	}

	options := &Options{
		TabWidth:  8,
		TabIndent: true,
		Comments:  true,
		Fragment:  true,
	}

	for _, tt := range tests {
		if *only != "" && tt.name != *only {
			continue
		}
		buf, err := Process(tt.name+".go", []byte(tt.in), options)
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

	pkgIndexOnce = sync.Once{}

	origStdlib := stdlib
	defer func() {
		stdlib = origStdlib
	}()
	stdlib = nil

	// Test against imaginary bits/bytes package in std lib
	bytesDir := filepath.Join(goroot, "src", "pkg", "bits", "bytes")
	for _, tag := range build.Default.ReleaseTags {
		// Go 1.4 rearranged the GOROOT tree to remove the "pkg" path component.
		if tag == "go1.4" {
			bytesDir = filepath.Join(goroot, "src", "bits", "bytes")
		}
	}
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

	got, rename, err := findImportGoPath("bytes", map[string]bool{"Buffer2": true}, "x.go")
	if err != nil {
		t.Fatal(err)
	}
	if got != bytesPkgPath || rename {
		t.Errorf(`findImportGoPath("bytes", Buffer2 ...)=%q, %t, want "%s", false`, got, rename, bytesPkgPath)
	}

	got, rename, err = findImportGoPath("bytes", map[string]bool{"Missing": true}, "x.go")
	if err != nil {
		t.Fatal(err)
	}
	if got != "" || rename {
		t.Errorf(`findImportGoPath("bytes", Missing ...)=%q, %t, want "", false`, got, rename)
	}
}

func TestFindImportInternal(t *testing.T) {
	pkgIndexOnce = sync.Once{}
	oldGOPATH := build.Default.GOPATH
	build.Default.GOPATH = ""
	defer func() {
		build.Default.GOPATH = oldGOPATH
	}()

	_, err := os.Stat(filepath.Join(runtime.GOROOT(), "src/internal"))
	if err != nil {
		t.Skip(err)
	}

	got, rename, err := findImportGoPath("race", map[string]bool{"Acquire": true}, filepath.Join(runtime.GOROOT(), "src/math/x.go"))
	if err != nil {
		t.Fatal(err)
	}
	if got != "internal/race" || rename {
		t.Errorf(`findImportGoPath("race", Acquire ...)=%q, %t, want "internal/race", false`, got, rename)
	}

	// should not be able to use internal from outside that tree
	got, rename, err = findImportGoPath("race", map[string]bool{"Acquire": true}, filepath.Join(runtime.GOROOT(), "x.go"))
	if err != nil {
		t.Fatal(err)
	}
	if got != "" || rename {
		t.Errorf(`findImportGoPath("race", Acquire ...)=%q, %t, want "", false`, got, rename)
	}
}

func TestFindImportVendor(t *testing.T) {
	pkgIndexOnce = sync.Once{}
	oldGOPATH := build.Default.GOPATH
	build.Default.GOPATH = ""
	defer func() {
		build.Default.GOPATH = oldGOPATH
	}()

	_, err := os.Stat(filepath.Join(runtime.GOROOT(), "src/vendor"))
	if err != nil {
		t.Skip(err)
	}

	got, rename, err := findImportGoPath("hpack", map[string]bool{"HuffmanDecode": true}, filepath.Join(runtime.GOROOT(), "src/math/x.go"))
	if err != nil {
		t.Fatal(err)
	}
	if got != "golang.org/x/net/http2/hpack" || rename {
		t.Errorf(`findImportGoPath("hpack", HuffmanDecode ...)=%q, %t, want "golang.org/x/net/http2/hpack", false`, got, rename)
	}

	// should not be able to use vendor from outside that tree
	got, rename, err = findImportGoPath("hpack", map[string]bool{"HuffmanDecode": true}, filepath.Join(runtime.GOROOT(), "x.go"))
	if err != nil {
		t.Fatal(err)
	}
	if got != "" || rename {
		t.Errorf(`findImportGoPath("hpack", HuffmanDecode ...)=%q, %t, want "", false`, got, rename)
	}
}

func TestProcessVendor(t *testing.T) {
	pkgIndexOnce = sync.Once{}
	oldGOPATH := build.Default.GOPATH
	build.Default.GOPATH = ""
	defer func() {
		build.Default.GOPATH = oldGOPATH
	}()

	_, err := os.Stat(filepath.Join(runtime.GOROOT(), "src/vendor"))
	if err != nil {
		t.Skip(err)
	}

	target := filepath.Join(runtime.GOROOT(), "src/math/x.go")
	out, err := Process(target, []byte("package http\nimport \"bytes\"\nfunc f() { strings.NewReader(); hpack.HuffmanDecode() }\n"), nil)

	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out, []byte("\"golang.org/x/net/http2/hpack\"")) {
		t.Fatalf("Process(%q) did not add expected hpack import:\n%s", target, out)
	}
}

func TestFindImportStdlib(t *testing.T) {
	tests := []struct {
		pkg     string
		symbols []string
		want    string
	}{
		{"http", []string{"Get"}, "net/http"},
		{"http", []string{"Get", "Post"}, "net/http"},
		{"http", []string{"Get", "Foo"}, ""},
		{"bytes", []string{"Buffer"}, "bytes"},
		{"ioutil", []string{"Discard"}, "io/ioutil"},
	}
	for _, tt := range tests {
		got, rename, ok := findImportStdlib(tt.pkg, strSet(tt.symbols))
		if (got != "") != ok {
			t.Error("findImportStdlib return value inconsistent")
		}
		if got != tt.want || rename {
			t.Errorf("findImportStdlib(%q, %q) = %q, %t; want %q, false", tt.pkg, tt.symbols, got, rename, tt.want)
		}
	}
}

func strSet(ss []string) map[string]bool {
	m := make(map[string]bool)
	for _, s := range ss {
		m[s] = true
	}
	return m
}
