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
	"strings"
	"sync"
	"testing"
)

var only = flag.String("only", "", "If non-empty, the fix test to run")

var tests = []struct {
	name       string
	formatOnly bool
	in, out    string
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

import (
	"bytes"
)

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

	// Unused named import is mistaken for unnamed import
	// golang.org/issue/8149
	{
		name: "issue 8149",
		in: `package main

import (
	"fmt"
	x "fmt"
)

func main() { fmt.Println() }
`,
		out: `package main

import (
	"fmt"
)

func main() { fmt.Println() }
`,
	},

	// FormatOnly
	{
		name:       "format only",
		formatOnly: true,
		in: `package main

import (
"fmt"
"golang.org/x/foo"
)

func main() {}
`,
		out: `package main

import (
	"fmt"

	"golang.org/x/foo"
)

func main() {}
`,
	},

	{
		name: "do not make grouped imports non-grouped",
		in: `package p

import (
	"bytes"
	"fmt"
)

var _ = fmt.Sprintf
`,
		out: `package p

import (
	"fmt"
)

var _ = fmt.Sprintf
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
		options.FormatOnly = tt.formatOnly
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

// Test support for packages in GOPATH that are actually symlinks.
// Also test that a symlink loop does not block the process.
func TestImportSymlinks(t *testing.T) {
	switch runtime.GOOS {
	case "windows", "plan9":
		t.Skipf("skipping test on %q as there are no symlinks", runtime.GOOS)
	}

	newGoPath, err := ioutil.TempDir("", "symlinktest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(newGoPath)

	// Create:
	//    $GOPATH/target/
	//    $GOPATH/target/f.go  // package mypkg\nvar Foo = 123\n
	//    $GOPATH/src/x/
	//    $GOPATH/src/x/mypkg => $GOPATH/target   // symlink
	//    $GOPATH/src/x/apkg  => $GOPATH/src/x    // symlink loop
	// Test:
	//    $GOPATH/src/myotherpkg/toformat.go referencing mypkg.Foo

	targetPath := newGoPath + "/target"
	if err := os.MkdirAll(targetPath, 0755); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(targetPath+"/f.go", []byte("package mypkg\nvar Foo = 123\n"), 0666); err != nil {
		t.Fatal(err)
	}

	symlinkPath := newGoPath + "/src/x/mypkg"
	if err := os.MkdirAll(filepath.Dir(symlinkPath), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(targetPath, symlinkPath); err != nil {
		t.Fatal(err)
	}

	// Add a symlink loop.
	if err := os.Symlink(newGoPath+"/src/x", newGoPath+"/src/x/apkg"); err != nil {
		t.Fatal(err)
	}

	withEmptyGoPath(func() {
		build.Default.GOPATH = newGoPath

		input := `package p

var (
	_ = fmt.Print
	_ = mypkg.Foo
)
`
		output := `package p

import (
	"fmt"
	"x/mypkg"
)

var (
	_ = fmt.Print
	_ = mypkg.Foo
)
`
		buf, err := Process(newGoPath+"/src/myotherpkg/toformat.go", []byte(input), &Options{})
		if err != nil {
			t.Fatal(err)
		}
		if got := string(buf); got != output {
			t.Fatalf("results differ\nGOT:\n%s\nWANT:\n%s\n", got, output)
		}
	})

	// Add a .goimportsignore and ensure it is respected.
	if err := ioutil.WriteFile(newGoPath+"/src/.goimportsignore", []byte("x/mypkg\n"), 0666); err != nil {
		t.Fatal(err)
	}

	withEmptyGoPath(func() {
		build.Default.GOPATH = newGoPath

		input := `package p

var (
	_ = fmt.Print
	_ = mypkg.Foo
)
`
		output := `package p

import "fmt"

var (
	_ = fmt.Print
	_ = mypkg.Foo
)
`
		buf, err := Process(newGoPath+"/src/myotherpkg/toformat.go", []byte(input), &Options{})
		if err != nil {
			t.Fatal(err)
		}
		if got := string(buf); got != output {
			t.Fatalf("ignored results differ\nGOT:\n%s\nWANT:\n%s\n", got, output)
		}
	})

}

// Test for correctly identifying the name of a vendored package when it
// differs from its directory name. In this test, the import line
// "mypkg.com/mypkg.v1" would be removed if goimports wasn't able to detect
// that the package name is "mypkg".
func TestFixImportsVendorPackage(t *testing.T) {
	// Skip this test on go versions with no vendor support.
	if _, err := os.Stat(filepath.Join(runtime.GOROOT(), "src/vendor")); err != nil {
		t.Skip(err)
	}
	testConfig{
		gopathFiles: map[string]string{
			"mypkg.com/outpkg/vendor/mypkg.com/mypkg.v1/f.go": "package mypkg\nvar Foo = 123\n",
		},
	}.test(t, func(t *goimportTest) {
		input := `package p

import (
	"fmt"

	"mypkg.com/mypkg.v1"
)

var (
	_ = fmt.Print
	_ = mypkg.Foo
)
`
		buf, err := Process(filepath.Join(t.gopath, "src/mypkg.com/outpkg/toformat.go"), []byte(input), &Options{})
		if err != nil {
			t.Fatal(err)
		}
		if got := string(buf); got != input {
			t.Fatalf("results differ\nGOT:\n%s\nWANT:\n%s\n", got, input)
		}
	})
}

func TestFindImportGoPath(t *testing.T) {
	goroot, err := ioutil.TempDir("", "goimports-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(goroot)

	origStdlib := stdlib
	defer func() {
		stdlib = origStdlib
	}()
	stdlib = nil

	withEmptyGoPath(func() {
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
		build.Default.GOROOT = goroot

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
	})
}

func init() {
	inTests = true
}

func withEmptyGoPath(fn func()) {
	testMu.Lock()

	dirScanMu.Lock()
	populateIgnoreOnce = sync.Once{}
	scanGoRootOnce = sync.Once{}
	scanGoPathOnce = sync.Once{}
	dirScan = nil
	ignoredDirs = nil
	scanGoRootDone = make(chan struct{})
	dirScanMu.Unlock()

	oldGOPATH := build.Default.GOPATH
	oldGOROOT := build.Default.GOROOT
	build.Default.GOPATH = ""
	testHookScanDir = func(string) {}
	testMu.Unlock()

	defer func() {
		testMu.Lock()
		testHookScanDir = func(string) {}
		build.Default.GOPATH = oldGOPATH
		build.Default.GOROOT = oldGOROOT
		testMu.Unlock()
	}()

	fn()
}

func TestFindImportInternal(t *testing.T) {
	withEmptyGoPath(func() {
		// Check for src/internal/race, not just src/internal,
		// so that we can run this test also against go1.5
		// (which doesn't contain that file).
		_, err := os.Stat(filepath.Join(runtime.GOROOT(), "src/internal/race"))
		if err != nil {
			t.Skip(err)
		}

		got, rename, err := findImportGoPath("race", map[string]bool{"Acquire": true}, filepath.Join(runtime.GOROOT(), "src/math/x.go"))
		if err != nil {
			t.Fatal(err)
		}
		if got != "internal/race" || rename {
			t.Errorf(`findImportGoPath("race", Acquire ...) = %q, %t; want "internal/race", false`, got, rename)
		}

		// should not be able to use internal from outside that tree
		got, rename, err = findImportGoPath("race", map[string]bool{"Acquire": true}, filepath.Join(runtime.GOROOT(), "x.go"))
		if err != nil {
			t.Fatal(err)
		}
		if got != "" || rename {
			t.Errorf(`findImportGoPath("race", Acquire ...)=%q, %t, want "", false`, got, rename)
		}
	})
}

// rand.Read should prefer crypto/rand.Read, not math/rand.Read.
func TestFindImportRandRead(t *testing.T) {
	withEmptyGoPath(func() {
		file := filepath.Join(runtime.GOROOT(), "src/foo/x.go") // dummy
		tests := []struct {
			syms []string
			want string
		}{
			{
				syms: []string{"Read"},
				want: "crypto/rand",
			},
			{
				syms: []string{"Read", "NewZipf"},
				want: "math/rand",
			},
			{
				syms: []string{"NewZipf"},
				want: "math/rand",
			},
			{
				syms: []string{"Read", "Prime"},
				want: "crypto/rand",
			},
		}
		for _, tt := range tests {
			m := map[string]bool{}
			for _, sym := range tt.syms {
				m[sym] = true
			}
			got, _, err := findImportGoPath("rand", m, file)
			if err != nil {
				t.Errorf("for %q: %v", tt.syms, err)
				continue
			}
			if got != tt.want {
				t.Errorf("for %q, findImportGoPath = %q; want %q", tt.syms, got, tt.want)
			}
		}
	})
}

func TestFindImportVendor(t *testing.T) {
	testConfig{
		gorootFiles: map[string]string{
			"vendor/golang.org/x/net/http2/hpack/huffman.go": "package hpack\nfunc HuffmanDecode() { }\n",
		},
	}.test(t, func(t *goimportTest) {
		got, rename, err := findImportGoPath("hpack", map[string]bool{"HuffmanDecode": true}, filepath.Join(t.goroot, "src/math/x.go"))
		if err != nil {
			t.Fatal(err)
		}
		want := "golang.org/x/net/http2/hpack"
		if got != want || rename {
			t.Errorf(`findImportGoPath("hpack", HuffmanDecode ...) = %q, %t; want %q, false`, got, rename, want)
		}
	})
}

func TestProcessVendor(t *testing.T) {
	withEmptyGoPath(func() {
		_, err := os.Stat(filepath.Join(runtime.GOROOT(), "src/vendor"))
		if err != nil {
			t.Skip(err)
		}

		target := filepath.Join(runtime.GOROOT(), "src/math/x.go")
		out, err := Process(target, []byte("package http\nimport \"bytes\"\nfunc f() { strings.NewReader(); hpack.HuffmanDecode() }\n"), nil)

		if err != nil {
			t.Fatal(err)
		}

		want := "golang_org/x/net/http2/hpack"
		if _, err := os.Stat(filepath.Join(runtime.GOROOT(), "src/vendor", want)); os.IsNotExist(err) {
			want = "golang.org/x/net/http2/hpack"
		}

		if !bytes.Contains(out, []byte(want)) {
			t.Fatalf("Process(%q) did not add expected hpack import %q; got:\n%s", target, want, out)
		}
	})
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

type testConfig struct {
	// goroot and gopath optionally specifies the path on disk
	// to use for the GOROOT and GOPATH. If empty, a temp directory
	// is made if needed.
	goroot, gopath string

	// gorootFiles optionally specifies the complete contents of GOROOT to use,
	// If nil, the normal current $GOROOT is used.
	gorootFiles map[string]string // paths relative to $GOROOT/src to contents

	// gopathFiles is like gorootFiles, but for $GOPATH.
	// If nil, there is no GOPATH, though.
	gopathFiles map[string]string // paths relative to $GOPATH/src to contents
}

func mustTempDir(t *testing.T, prefix string) string {
	dir, err := ioutil.TempDir("", prefix)
	if err != nil {
		t.Fatal(err)
	}
	return dir
}

func mapToDir(destDir string, files map[string]string) error {
	for path, contents := range files {
		file := filepath.Join(destDir, "src", path)
		if err := os.MkdirAll(filepath.Dir(file), 0755); err != nil {
			return err
		}
		var err error
		if strings.HasPrefix(contents, "LINK:") {
			err = os.Symlink(strings.TrimPrefix(contents, "LINK:"), file)
		} else {
			err = ioutil.WriteFile(file, []byte(contents), 0644)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func (c testConfig) test(t *testing.T, fn func(*goimportTest)) {
	goroot := c.goroot
	gopath := c.gopath

	if c.gorootFiles != nil && goroot == "" {
		goroot = mustTempDir(t, "goroot-")
		defer os.RemoveAll(goroot)
	}
	if err := mapToDir(goroot, c.gorootFiles); err != nil {
		t.Fatal(err)
	}

	if c.gopathFiles != nil && gopath == "" {
		gopath = mustTempDir(t, "gopath-")
		defer os.RemoveAll(gopath)
	}
	if err := mapToDir(gopath, c.gopathFiles); err != nil {
		t.Fatal(err)
	}

	withEmptyGoPath(func() {
		if goroot != "" {
			build.Default.GOROOT = goroot
		}
		build.Default.GOPATH = gopath

		it := &goimportTest{
			T:      t,
			goroot: build.Default.GOROOT,
			gopath: gopath,
			ctx:    &build.Default,
		}
		fn(it)
	})
}

type goimportTest struct {
	*testing.T
	ctx    *build.Context
	goroot string
	gopath string
}

// Tests that added imports are renamed when the import path's base doesn't
// match its package name. For example, we want to generate:
//
//     import cloudbilling "google.golang.org/api/cloudbilling/v1"
func TestRenameWhenPackageNameMismatch(t *testing.T) {
	testConfig{
		gopathFiles: map[string]string{
			"foo/bar/v1/x.go": "package bar \n const X = 1",
		},
	}.test(t, func(t *goimportTest) {
		buf, err := Process(t.gopath+"/src/test/t.go", []byte("package main \n const Y = bar.X"), &Options{})
		if err != nil {
			t.Fatal(err)
		}
		const want = `package main

import bar "foo/bar/v1"

const Y = bar.X
`
		if string(buf) != want {
			t.Errorf("Got:\n%s\nWant:\n%s", buf, want)
		}
	})
}

// Tests that the LocalPrefix option causes imports
// to be added into a later group (num=3).
func TestLocalPrefix(t *testing.T) {
	defer func(s string) { LocalPrefix = s }(LocalPrefix)
	LocalPrefix = "foo/"

	testConfig{
		gopathFiles: map[string]string{
			"foo/bar/bar.go": "package bar \n const X = 1",
		},
	}.test(t, func(t *goimportTest) {
		buf, err := Process(t.gopath+"/src/test/t.go", []byte("package main \n const Y = bar.X \n const _ = runtime.GOOS"), &Options{})
		if err != nil {
			t.Fatal(err)
		}
		const want = `package main

import (
	"runtime"

	"foo/bar"
)

const Y = bar.X
const _ = runtime.GOOS
`
		if string(buf) != want {
			t.Errorf("Got:\n%s\nWant:\n%s", buf, want)
		}
	})
}

// Tests that running goimport on files in GOROOT (for people hacking
// on Go itself) don't cause the GOPATH to be scanned (which might be
// much bigger).
func TestOptimizationWhenInGoroot(t *testing.T) {
	testConfig{
		gopathFiles: map[string]string{
			"foo/foo.go": "package foo\nconst X = 1\n",
		},
	}.test(t, func(t *goimportTest) {
		testHookScanDir = func(dir string) {
			if dir != filepath.Join(build.Default.GOROOT, "src") {
				t.Errorf("unexpected dir scan of %s", dir)
			}
		}
		const in = "package foo\n\nconst Y = bar.X\n"
		buf, err := Process(t.goroot+"/src/foo/foo.go", []byte(in), nil)
		if err != nil {
			t.Fatal(err)
		}
		if string(buf) != in {
			t.Errorf("got:\n%q\nwant unchanged:\n%q\n", in, buf)
		}
	})
}

// Tests that "package documentation" files are ignored.
func TestIgnoreDocumentationPackage(t *testing.T) {
	testConfig{
		gopathFiles: map[string]string{
			"foo/foo.go": "package foo\nconst X = 1\n",
			"foo/doc.go": "package documentation \n // just to confuse things\n",
		},
	}.test(t, func(t *goimportTest) {
		const in = "package x\n\nconst Y = foo.X\n"
		const want = "package x\n\nimport \"foo\"\n\nconst Y = foo.X\n"
		buf, err := Process(t.gopath+"/src/x/x.go", []byte(in), nil)
		if err != nil {
			t.Fatal(err)
		}
		if string(buf) != want {
			t.Errorf("wrong output.\ngot:\n%q\nwant:\n%q\n", in, want)
		}
	})
}

// Tests importPathToNameGoPathParse and in particular that it stops
// after finding the first non-documentation package name, not
// reporting an error on inconsistent package names (since it should
// never make it that far).
func TestImportPathToNameGoPathParse(t *testing.T) {
	testConfig{
		gopathFiles: map[string]string{
			"example.net/pkg/doc.go": "package documentation\n", // ignored
			"example.net/pkg/gen.go": "package main\n",          // also ignored
			"example.net/pkg/pkg.go": "package the_pkg_name_to_find\n  and this syntax error is ignored because of parser.PackageClauseOnly",
			"example.net/pkg/z.go":   "package inconsistent\n", // inconsistent but ignored
		},
	}.test(t, func(t *goimportTest) {
		got, err := importPathToNameGoPathParse("example.net/pkg", filepath.Join(t.gopath, "src", "other.net"))
		if err != nil {
			t.Fatal(err)
		}
		const want = "the_pkg_name_to_find"
		if got != want {
			t.Errorf("importPathToNameGoPathParse(..) = %q; want %q", got, want)
		}
	})
}

func TestIgnoreConfiguration(t *testing.T) {
	testConfig{
		gopathFiles: map[string]string{
			".goimportsignore":                                     "# comment line\n\n example.net", // tests comment, blank line, whitespace trimming
			"example.net/pkg/pkg.go":                               "package pkg\nconst X = 1",
			"otherwise-longer-so-worse.example.net/foo/pkg/pkg.go": "package pkg\nconst X = 1",
		},
	}.test(t, func(t *goimportTest) {
		const in = "package x\n\nconst _ = pkg.X\n"
		const want = "package x\n\nimport \"otherwise-longer-so-worse.example.net/foo/pkg\"\n\nconst _ = pkg.X\n"
		buf, err := Process(t.gopath+"/src/x/x.go", []byte(in), nil)
		if err != nil {
			t.Fatal(err)
		}
		if string(buf) != want {
			t.Errorf("wrong output.\ngot:\n%q\nwant:\n%q\n", buf, want)
		}
	})
}

// Skip "node_modules" directory.
func TestSkipNodeModules(t *testing.T) {
	testConfig{
		gopathFiles: map[string]string{
			"example.net/node_modules/pkg/a.go":         "package pkg\nconst X = 1",
			"otherwise-longer.net/not_modules/pkg/a.go": "package pkg\nconst X = 1",
		},
	}.test(t, func(t *goimportTest) {
		const in = "package x\n\nconst _ = pkg.X\n"
		const want = "package x\n\nimport \"otherwise-longer.net/not_modules/pkg\"\n\nconst _ = pkg.X\n"
		buf, err := Process(t.gopath+"/src/x/x.go", []byte(in), nil)
		if err != nil {
			t.Fatal(err)
		}
		if string(buf) != want {
			t.Errorf("wrong output.\ngot:\n%q\nwant:\n%q\n", buf, want)
		}
	})
}

// golang.org/issue/16458 -- if GOROOT is a prefix of GOPATH, GOPATH is ignored.
func TestGoRootPrefixOfGoPath(t *testing.T) {
	dir := mustTempDir(t, "importstest")
	defer os.RemoveAll(dir)
	testConfig{
		goroot: filepath.Join(dir, "go"),
		gopath: filepath.Join(dir, "gopath"),
		gopathFiles: map[string]string{
			"example.com/foo/pkg.go": "package foo\nconst X = 1",
		},
	}.test(t, func(t *goimportTest) {
		const in = "package x\n\nconst _ = foo.X\n"
		const want = "package x\n\nimport \"example.com/foo\"\n\nconst _ = foo.X\n"
		buf, err := Process(t.gopath+"/src/x/x.go", []byte(in), nil)
		if err != nil {
			t.Fatal(err)
		}
		if string(buf) != want {
			t.Errorf("wrong output.\ngot:\n%q\nwant:\n%q\n", buf, want)
		}
	})

}

const testGlobalImportsUsesGlobal = `package globalimporttest

func doSomething() {
	t := time.Now()
}
`

const testGlobalImportsGlobalDecl = `package globalimporttest

type Time struct{}

func (t Time) Now() Time {
	return Time{}
}

var time Time
`

// Tests that package global variables with the same name and function name as
// a function in a separate package do not result in an import which masks
// the global variable
func TestGlobalImports(t *testing.T) {
	const pkg = "globalimporttest"
	const usesGlobalFile = pkg + "/uses_global.go"
	testConfig{
		gopathFiles: map[string]string{
			usesGlobalFile:     testGlobalImportsUsesGlobal,
			pkg + "/global.go": testGlobalImportsGlobalDecl,
		},
	}.test(t, func(t *goimportTest) {
		buf, err := Process(
			t.gopath+"/src/"+usesGlobalFile, []byte(testGlobalImportsUsesGlobal), nil)
		if err != nil {
			t.Fatal(err)
		}
		if string(buf) != testGlobalImportsUsesGlobal {
			t.Errorf("wrong output.\ngot:\n%q\nwant:\n%q\n", buf, testGlobalImportsUsesGlobal)
		}
	})
}

// Tests that sibling files - other files in the same package - can provide an
// import that may not be the default one otherwise.
func TestSiblingImports(t *testing.T) {

	// provide is the sibling file that provides the desired import.
	const provide = `package siblingimporttest

import "local/log"

func LogSomething() {
	log.Print("Something")
}
`

	// need is the file being tested that needs the import.
	const need = `package siblingimporttest

func LogSomethingElse() {
	log.Print("Something else")
}
`

	// want is the expected result file
	const want = `package siblingimporttest

import "local/log"

func LogSomethingElse() {
	log.Print("Something else")
}
`

	const pkg = "siblingimporttest"
	const siblingFile = pkg + "/needs_import.go"
	testConfig{
		gopathFiles: map[string]string{
			siblingFile:                 need,
			pkg + "/provides_import.go": provide,
		},
	}.test(t, func(t *goimportTest) {
		buf, err := Process(
			t.gopath+"/src/"+siblingFile, []byte(need), nil)
		if err != nil {
			t.Fatal(err)
		}
		if string(buf) != want {
			t.Errorf("wrong output.\ngot:\n%q\nwant:\n%q\n", buf, want)
		}
	})
}

func strSet(ss []string) map[string]bool {
	m := make(map[string]bool)
	for _, s := range ss {
		m[s] = true
	}
	return m
}

func TestPkgIsCandidate(t *testing.T) {
	tests := [...]struct {
		filename string
		pkgIdent string
		pkg      *pkg
		want     bool
	}{
		// normal match
		0: {
			filename: "/gopath/src/my/pkg/pkg.go",
			pkgIdent: "client",
			pkg: &pkg{
				dir:             "/gopath/src/client",
				importPath:      "client",
				importPathShort: "client",
			},
			want: true,
		},
		// not a match
		1: {
			filename: "/gopath/src/my/pkg/pkg.go",
			pkgIdent: "zzz",
			pkg: &pkg{
				dir:             "/gopath/src/client",
				importPath:      "client",
				importPathShort: "client",
			},
			want: false,
		},
		// would be a match, but "client" appears too deep.
		2: {
			filename: "/gopath/src/my/pkg/pkg.go",
			pkgIdent: "client",
			pkg: &pkg{
				dir:             "/gopath/src/client/foo/foo/foo",
				importPath:      "client/foo/foo",
				importPathShort: "client/foo/foo",
			},
			want: false,
		},
		// not an exact match, but substring is good enough.
		3: {
			filename: "/gopath/src/my/pkg/pkg.go",
			pkgIdent: "client",
			pkg: &pkg{
				dir:             "/gopath/src/foo/go-client",
				importPath:      "foo/go-client",
				importPathShort: "foo/go-client",
			},
			want: true,
		},
		// "internal" package, and not visible
		4: {
			filename: "/gopath/src/my/pkg/pkg.go",
			pkgIdent: "client",
			pkg: &pkg{
				dir:             "/gopath/src/foo/internal/client",
				importPath:      "foo/internal/client",
				importPathShort: "foo/internal/client",
			},
			want: false,
		},
		// "internal" package but visible
		5: {
			filename: "/gopath/src/foo/bar.go",
			pkgIdent: "client",
			pkg: &pkg{
				dir:             "/gopath/src/foo/internal/client",
				importPath:      "foo/internal/client",
				importPathShort: "foo/internal/client",
			},
			want: true,
		},
		// "vendor" package not visible
		6: {
			filename: "/gopath/src/foo/bar.go",
			pkgIdent: "client",
			pkg: &pkg{
				dir:             "/gopath/src/other/vendor/client",
				importPath:      "other/vendor/client",
				importPathShort: "client",
			},
			want: false,
		},
		// "vendor" package, visible
		7: {
			filename: "/gopath/src/foo/bar.go",
			pkgIdent: "client",
			pkg: &pkg{
				dir:             "/gopath/src/foo/vendor/client",
				importPath:      "other/foo/client",
				importPathShort: "client",
			},
			want: true,
		},
		// Ignore hyphens.
		8: {
			filename: "/gopath/src/foo/bar.go",
			pkgIdent: "socketio",
			pkg: &pkg{
				dir:             "/gopath/src/foo/socket-io",
				importPath:      "foo/socket-io",
				importPathShort: "foo/socket-io",
			},
			want: true,
		},
		// Ignore case.
		9: {
			filename: "/gopath/src/foo/bar.go",
			pkgIdent: "fooprod",
			pkg: &pkg{
				dir:             "/gopath/src/foo/FooPROD",
				importPath:      "foo/FooPROD",
				importPathShort: "foo/FooPROD",
			},
			want: true,
		},
		// Ignoring both hyphens and case together.
		10: {
			filename: "/gopath/src/foo/bar.go",
			pkgIdent: "fooprod",
			pkg: &pkg{
				dir:             "/gopath/src/foo/Foo-PROD",
				importPath:      "foo/Foo-PROD",
				importPathShort: "foo/Foo-PROD",
			},
			want: true,
		},
	}
	for i, tt := range tests {
		got := pkgIsCandidate(tt.filename, tt.pkgIdent, tt.pkg)
		if got != tt.want {
			t.Errorf("test %d. pkgIsCandidate(%q, %q, %+v) = %v; want %v",
				i, tt.filename, tt.pkgIdent, *tt.pkg, got, tt.want)
		}
	}
}

func TestShouldTraverse(t *testing.T) {
	switch runtime.GOOS {
	case "windows", "plan9":
		t.Skipf("skipping symlink-requiring test on %s", runtime.GOOS)
	}

	dir, err := ioutil.TempDir("", "goimports-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	// Note: mapToDir prepends "src" to each element, since
	// mapToDir was made for creating GOPATHs.
	if err := mapToDir(dir, map[string]string{
		"foo/foo2/file.txt":        "",
		"foo/foo2/link-to-src":     "LINK:" + dir + "/src",
		"foo/foo2/link-to-src-foo": "LINK:" + dir + "/src/foo",
		"foo/foo2/link-to-dot":     "LINK:.",
		"bar/bar2/file.txt":        "",
		"bar/bar2/link-to-src-foo": "LINK:" + dir + "/src/foo",

		"a/b/c": "LINK:" + dir + "/src/a/d",
		"a/d/e": "LINK:" + dir + "/src/a/b",
	}); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		dir  string
		file string
		want bool
	}{
		{
			dir:  dir + "/src/foo/foo2",
			file: "link-to-src-foo",
			want: false, // loop
		},
		{
			dir:  dir + "/src/foo/foo2",
			file: "link-to-src",
			want: false, // loop
		},
		{
			dir:  dir + "/src/foo/foo2",
			file: "link-to-dot",
			want: false, // loop
		},
		{
			dir:  dir + "/src/bar/bar2",
			file: "link-to-src-foo",
			want: true, // not a loop
		},
		{
			dir:  dir + "/src/a/b/c",
			file: "e",
			want: false, // loop: "e" is the same as "b".
		},
	}
	for i, tt := range tests {
		fi, err := os.Stat(filepath.Join(tt.dir, tt.file))
		if err != nil {
			t.Errorf("%d. Stat = %v", i, err)
			continue
		}
		got := shouldTraverse(tt.dir, fi)
		if got != tt.want {
			t.Errorf("%d. shouldTraverse(%q, %q) = %v; want %v", i, tt.dir, tt.file, got, tt.want)
		}
	}
}
