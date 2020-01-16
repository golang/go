// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"context"
	"flag"
	"fmt"
	"go/build"
	"log"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
)

var testDebug = flag.Bool("debug", false, "enable debug output")

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

  "github.com/golang/snappy"
)
func bar() {
var b bytes.Buffer
_ = snappy.ErrCorrupt
fmt.Println(b.String())
}
`,
		out: `package foo

import (
	"bytes"
	"fmt"

	"github.com/golang/snappy"
)

func bar() {
	var b bytes.Buffer
	_ = snappy.ErrCorrupt
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

  "github.com/golang/snappy"
)
func bar() {
_ = math.NaN
_ = fmt.Sprintf
_ = snappy.ErrCorrupt
}
`,
		out: `package foo

import (
	"fmt"
	"math"

	"github.com/golang/snappy"
)

func bar() {
	_ = math.NaN
	_ = fmt.Sprintf
	_ = snappy.ErrCorrupt
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

	// Make sure we don't add packages that don't have the right exports
	{
		name: "no_mismatched_add",
		in: `package foo

func bar() {
	_ := bytes.NonexistentSymbol
}
`,
		out: `package foo

func bar() {
	_ := bytes.NonexistentSymbol
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
_, _ = snappy.ErrCorrupt, p.P
}
`,
		out: `package foo

import (
	"fmt"
	"os"

	"github.com/golang/snappy"
	"rsc.io/p"
)

func foo() {
	_, _ = os.Args, fmt.Println
	_, _ = snappy.ErrCorrupt, p.P
}
`,
	},
	// Merge import blocks, even when no additions are required.
	{
		name: "merge_import_blocks_no_fix",
		in: `package foo

import (
	"fmt"
)
import "os"

import (
	"rsc.io/p"
)

var _, _ = os.Args, fmt.Println
var _, _ = snappy.ErrCorrupt, p.P
`,
		out: `package foo

import (
	"fmt"
	"os"

	"github.com/golang/snappy"
	"rsc.io/p"
)

var _, _ = os.Args, fmt.Println
var _, _ = snappy.ErrCorrupt, p.P
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
	_ = snappy.ErrCorrupt
}
`,
		out: `package foo

import (
	"fmt"
	"net"

	"github.com/golang/snappy"
)

func f() {
	_ = net.Dial
	_ = fmt.Printf
	_ = snappy.ErrCorrupt
}
`,
	},

	// Blank line between standard library and third-party stuff.
	{
		name: "blank_line_separating_std_and_third_party",
		in: `package foo

import (
	"github.com/golang/snappy"
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

	"github.com/golang/snappy"
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
		name: "new_imports_before_comment",
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
		name: "new_section_for_dotless_import",
		in: `package main

import (
"fmt"

"gu"
"manypackages.com/packagea"
)

var (
a = packagea.A
b = gu.A
c = fmt.Printf
)
`,
		out: `package main

import (
	"fmt"

	"gu"

	"manypackages.com/packagea"
)

var (
	a = packagea.A
	b = gu.A
	c = fmt.Printf
)
`,
	},

	{
		name: "fragment_with_main",
		in:   `func main(){fmt.Println("Hello, world")}`,
		out: `package main

import "fmt"

func main() { fmt.Println("Hello, world") }
`,
	},

	{
		name: "fragment_without_main",
		in:   `func notmain(){fmt.Println("Hello, world")}`,
		out: `import "fmt"

func notmain() { fmt.Println("Hello, world") }`,
	},

	// Remove first import within in a 2nd/3rd/4th/etc. section.
	// golang.org/issue/7679
	{
		name: "remove_first_import_in_section",
		in: `package main

import (
	"fmt"

	"manypackages.com/packagea"
	"manypackages.com/packageb"
)

func main() {
	var _ = fmt.Println
	//var _ = packagea.A
	var _ = packageb.B
}
`,
		out: `package main

import (
	"fmt"

	"manypackages.com/packageb"
)

func main() {
	var _ = fmt.Println
	//var _ = packagea.A
	var _ = packageb.B
}
`,
	},

	// Blank line can be added before all types of import declarations.
	// golang.org/issue/7866
	{
		name: "new_section_for_all_kinds_of_imports",
		in: `package main

import (
	"fmt"
	renamed_packagea "manypackages.com/packagea"

	. "manypackages.com/packageb"
	"io"

	_ "manypackages.com/packagec"
	"strings"
)

var _, _, _, _, _ = fmt.Errorf, io.Copy, strings.Contains, renamed_packagea.A, B
`,
		out: `package main

import (
	"fmt"

	renamed_packagea "manypackages.com/packagea"

	"io"

	. "manypackages.com/packageb"

	"strings"

	_ "manypackages.com/packagec"
)

var _, _, _, _, _ = fmt.Errorf, io.Copy, strings.Contains, renamed_packagea.A, B
`,
	},

	// Non-idempotent comment formatting
	// golang.org/issue/8035
	{
		name: "comments_formatted",
		in: `package main

import (
	"fmt"                     // A
	"go/ast"                  // B
	_ "manypackages.com/packagec"    // C
)

func main() { _, _ = fmt.Print, ast.Walk }
`,
		out: `package main

import (
	"fmt"    // A
	"go/ast" // B

	_ "manypackages.com/packagec" // C
)

func main() { _, _ = fmt.Print, ast.Walk }
`,
	},

	// Failure to delete all duplicate imports
	// golang.org/issue/8459
	{
		name: "remove_duplicates",
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
		name: "no_extra_groups",
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
		name: "named_import_doesnt_provide_package_name",
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
		name: "unused_named_import_removed",
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

	{
		name: "ignore_unexported_identifier",
		in: `package main
var _ = fmt.unexported`,
		out: `package main

var _ = fmt.unexported
`,
	},

	// FormatOnly
	{
		name:       "formatonly_works",
		formatOnly: true,
		in: `package main

import (
"fmt"
"manypackages.com/packagea"
)

func main() {}
`,
		out: `package main

import (
	"fmt"

	"manypackages.com/packagea"
)

func main() {}
`,
	},

	{
		name: "preserve_import_group",
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

	{
		name: "import_grouping_not_path_dependent_no_groups",
		in: `package main

import (
	"time"
)

func main() {
	_ = snappy.ErrCorrupt
	_ = p.P
	_ = time.Parse
}
`,
		out: `package main

import (
	"time"

	"github.com/golang/snappy"
	"rsc.io/p"
)

func main() {
	_ = snappy.ErrCorrupt
	_ = p.P
	_ = time.Parse
}
`,
	},

	{
		name: "import_grouping_not_path_dependent_existing_group",
		in: `package main

import (
	"time"

	"github.com/golang/snappy"
)

func main() {
	_ = snappy.ErrCorrupt
	_ = p.P
	_ = time.Parse
}
`,
		out: `package main

import (
	"time"

	"github.com/golang/snappy"
	"rsc.io/p"
)

func main() {
	_ = snappy.ErrCorrupt
	_ = p.P
	_ = time.Parse
}
`,
	},

	// golang.org/issue/12097
	{
		name: "package_statement_insertion_preserves_comments",
		in: `// a
// b
// c

func main() {
    _ = fmt.Println
}`,
		out: `package main

import "fmt"

// a
// b
// c

func main() {
	_ = fmt.Println
}
`,
	},

	{
		name: "import_comment_stays_on_import",
		in: `package main

import (
	"math" // fun
)

func main() {
	x := math.MaxInt64
	fmt.Println(strings.Join(",", []string{"hi"}), x)
}`,
		out: `package main

import (
	"fmt"
	"math" // fun
	"strings"
)

func main() {
	x := math.MaxInt64
	fmt.Println(strings.Join(",", []string{"hi"}), x)
}
`,
	},

	{
		name: "no_blank_after_comment",
		in: `package main

import (
	_ "io"
	_ "net/http"
	_ "net/http/pprof" // install the pprof http handlers
	_ "strings"
)

func main() {
}
`,
		out: `package main

import (
	_ "io"
	_ "net/http"
	_ "net/http/pprof" // install the pprof http handlers
	_ "strings"
)

func main() {
}
`,
	},

	{
		name: "no_blank_after_comment_reordered",
		in: `package main

import (
	_ "io"
	_ "net/http/pprof" // install the pprof http handlers
	_ "net/http"
	_ "strings"
)

func main() {
}
`,
		out: `package main

import (
	_ "io"
	_ "net/http"
	_ "net/http/pprof" // install the pprof http handlers
	_ "strings"
)

func main() {
}
`,
	},

	{
		name: "no_blank_after_comment_unnamed",
		in: `package main

import (
	"encoding/json"
	"io"
	"net/http"
	_ "net/http/pprof" // install the pprof http handlers
	"strings"

	"manypackages.com/packagea"
)

func main() {
	_ = strings.ToUpper("hello")
	_ = io.EOF
	var (
		_ json.Number
		_ *http.Request
		_ packagea.A
	)
}
`,
		out: `package main

import (
	"encoding/json"
	"io"
	"net/http"
	_ "net/http/pprof" // install the pprof http handlers
	"strings"

	"manypackages.com/packagea"
)

func main() {
	_ = strings.ToUpper("hello")
	_ = io.EOF
	var (
		_ json.Number
		_ *http.Request
		_ packagea.A
	)
}
`,
	},

	{
		name: "blank_after_package_statement_with_comment",
		in: `package p // comment

import "math"

var _ = fmt.Printf
`,
		out: `package p // comment

import "fmt"

var _ = fmt.Printf
`,
	},

	{
		name: "blank_after_package_statement_no_comment",
		in: `package p

import "math"

var _ = fmt.Printf
`,
		out: `package p

import "fmt"

var _ = fmt.Printf
`,
	},

	{
		name: "cryptorand_preferred_easy_possible",
		in: `package p

var _ = rand.Read
`,
		out: `package p

import "crypto/rand"

var _ = rand.Read
`,
	},

	{
		name: "cryptorand_preferred_easy_impossible",
		in: `package p

var _ = rand.NewZipf
`,
		out: `package p

import "math/rand"

var _ = rand.NewZipf
`,
	},

	{
		name: "cryptorand_preferred_complex_possible",
		in: `package p

var _, _ = rand.Read, rand.Prime
`,
		out: `package p

import "crypto/rand"

var _, _ = rand.Read, rand.Prime
`,
	},

	{
		name: "cryptorand_preferred_complex_impossible",
		in: `package p

var _, _ = rand.Read, rand.NewZipf
`,
		out: `package p

import "math/rand"

var _, _ = rand.Read, rand.NewZipf
`,
	},
}

func TestSimpleCases(t *testing.T) {
	const localPrefix = "local.com,github.com/local"
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			options := &Options{
				TabWidth:   8,
				TabIndent:  true,
				Comments:   true,
				Fragment:   true,
				FormatOnly: tt.formatOnly,
			}
			testConfig{
				modules: []packagestest.Module{
					{
						Name:  "golang.org/fake",
						Files: fm{"x.go": tt.in},
					},
					// Skeleton non-stdlib packages for use during testing.
					// Each includes one arbitrary symbol, e.g. the first declaration in the first file.
					// Try not to add more without a good reason.
					// DO NOT USE PACKAGES NOT LISTED HERE -- they will be downloaded!
					{
						Name:  "rsc.io",
						Files: fm{"p/x.go": "package p\nfunc P(){}\n"},
					},
					{
						Name:  "github.com/golang/snappy",
						Files: fm{"x.go": "package snappy\nvar ErrCorrupt error\n"},
					},
					{
						Name: "manypackages.com",
						Files: fm{
							"packagea/x.go": "package packagea\nfunc A(){}\n",
							"packageb/x.go": "package packageb\nfunc B(){}\n",
							"packagec/x.go": "package packagec\nfunc C(){}\n",
							"packaged/x.go": "package packaged\nfunc D(){}\n",
						},
					},
					{
						Name:  "local.com",
						Files: fm{"foo/x.go": "package foo\nfunc Foo(){}\n"},
					},
					{
						Name:  "github.com/local",
						Files: fm{"bar/x.go": "package bar\nfunc Bar(){}\n"},
					},
				},
			}.test(t, func(t *goimportTest) {
				t.env.LocalPrefix = localPrefix
				t.assertProcessEquals("golang.org/fake", "x.go", nil, options, tt.out)
			})

		})
	}
}

func TestAppengine(t *testing.T) {
	const input = `package p

var _, _, _ = fmt.Printf, appengine.Main, datastore.ErrInvalidEntityType
`

	const want = `package p

import (
	"fmt"

	"appengine"
	"appengine/datastore"
)

var _, _, _ = fmt.Printf, appengine.Main, datastore.ErrInvalidEntityType
`

	testConfig{
		gopathOnly: true, // can't create a module named appengine, so no module tests.
		modules: []packagestest.Module{
			{
				Name:  "golang.org/fake",
				Files: fm{"x.go": input},
			},
			{
				Name: "appengine",
				Files: fm{
					"x.go":           "package appengine\nfunc Main(){}\n",
					"datastore/x.go": "package datastore\nvar ErrInvalidEntityType error\n",
				},
			},
		},
	}.processTest(t, "golang.org/fake", "x.go", nil, nil, want)
}

func TestReadFromFilesystem(t *testing.T) {
	tests := []struct {
		name    string
		in, out string
	}{
		{
			name: "works",
			in: `package foo
func bar() {
fmt.Println("hi")
}
`,
			out: `package foo

import "fmt"

func bar() {
	fmt.Println("hi")
}
`,
		},
		{
			name: "missing_package",
			in: `
func bar() {
fmt.Println("hi")
}
`,
			out: `
import "fmt"

func bar() {
	fmt.Println("hi")
}
`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			options := &Options{
				TabWidth:  8,
				TabIndent: true,
				Comments:  true,
				Fragment:  true,
			}
			testConfig{
				module: packagestest.Module{
					Name:  "golang.org/fake",
					Files: fm{"x.go": tt.in},
				},
			}.processTest(t, "golang.org/fake", "x.go", nil, options, tt.out)
		})
	}

}

// Test support for packages in GOPATH that are actually symlinks.
// Also test that a symlink loop does not block the process.
func TestImportSymlinks(t *testing.T) {
	switch runtime.GOOS {
	case "windows", "plan9":
		t.Skipf("skipping test on %q as there are no symlinks", runtime.GOOS)
	}

	const input = `package p

var (
	_ = fmt.Print
	_ = mypkg.Foo
)
`
	const want = `package p

import (
	"fmt"

	"golang.org/fake/x/y/mypkg"
)

var (
	_ = fmt.Print
	_ = mypkg.Foo
)
`

	testConfig{
		module: packagestest.Module{
			Name: "golang.org/fake",
			Files: fm{
				"target/f.go":                "package mypkg\nvar Foo = 123\n",
				"x/y/mypkg":                  packagestest.Symlink("../../target"), // valid symlink
				"x/y/apkg":                   packagestest.Symlink(".."),           // symlink loop
				"myotherpackage/toformat.go": input,
			},
		},
	}.processTest(t, "golang.org/fake", "myotherpackage/toformat.go", nil, nil, want)
}

func TestImportSymlinksWithIgnore(t *testing.T) {
	switch runtime.GOOS {
	case "windows", "plan9":
		t.Skipf("skipping test on %q as there are no symlinks", runtime.GOOS)
	}

	const input = `package p

var (
	_ = fmt.Print
	_ = mypkg.Foo
)
`
	const want = `package p

import "fmt"

var (
	_ = fmt.Print
	_ = mypkg.Foo
)
`

	testConfig{
		gopathOnly: true,
		module: packagestest.Module{
			Name: "golang.org/fake",
			Files: fm{
				"target/f.go":            "package mypkg\nvar Foo = 123\n",
				"x/y/mypkg":              packagestest.Symlink("../../target"), // valid symlink
				"x/y/apkg":               packagestest.Symlink(".."),           // symlink loop
				"myotherpkg/toformat.go": input,
				"../../.goimportsignore": "golang.org/fake/x/y/mypkg\n",
			},
		},
	}.processTest(t, "golang.org/fake", "myotherpkg/toformat.go", nil, nil, want)
}

// Test for x/y/v2 convention for package y.
func TestModuleVersion(t *testing.T) {
	const input = `package p

import (
	"fmt"

	"github.com/foo/v2"
)

var (
	_ = fmt.Print
	_ = foo.Foo
)
`

	testConfig{
		modules: []packagestest.Module{
			{
				Name:  "mypkg.com/outpkg",
				Files: fm{"toformat.go": input},
			},
			{
				Name:  "github.com/foo/v2",
				Files: fm{"x.go": "package foo\n func Foo(){}\n"},
			},
		},
	}.processTest(t, "mypkg.com/outpkg", "toformat.go", nil, nil, input)
}

// Test for correctly identifying the name of a vendored package when it
// differs from its directory name. In this test, the import line
// "mypkg.com/mypkg_v1" would be removed if goimports wasn't able to detect
// that the package name is "mypkg".
func TestVendorPackage(t *testing.T) {
	const input = `package p
import (
	"fmt"
	"mypkg.com/mypkg_v1"
)
var _, _ = fmt.Print, mypkg.Foo
`

	const want = `package p

import (
	"fmt"

	mypkg "mypkg.com/mypkg_v1"
)

var _, _ = fmt.Print, mypkg.Foo
`

	testConfig{
		gopathOnly: true,
		module: packagestest.Module{
			Name: "mypkg.com/outpkg",
			Files: fm{
				"vendor/mypkg.com/mypkg_v1/f.go": "package mypkg\nvar Foo = 123\n",
				"toformat.go":                    input,
			},
		},
	}.processTest(t, "mypkg.com/outpkg", "toformat.go", nil, nil, want)
}

func TestInternal(t *testing.T) {
	const input = `package bar

var _ = race.Acquire
`
	const importAdded = `package bar

import "foo.com/internal/race"

var _ = race.Acquire
`

	// Packages under the same directory should be able to use internal packages.
	testConfig{
		module: packagestest.Module{
			Name: "foo.com",
			Files: fm{
				"internal/race/x.go": "package race\n func Acquire(){}\n",
				"bar/x.go":           input,
			},
		},
	}.processTest(t, "foo.com", "bar/x.go", nil, nil, importAdded)

	// Packages outside the same directory should not.
	testConfig{
		modules: []packagestest.Module{
			{
				Name:  "foo.com",
				Files: fm{"internal/race/x.go": "package race\n func Acquire(){}\n"},
			},
			{
				Name:  "bar.com",
				Files: fm{"x.go": input},
			},
		},
	}.processTest(t, "bar.com", "x.go", nil, nil, input)
}

func TestProcessVendor(t *testing.T) {
	const input = `package p

var _ = hpack.HuffmanDecode
`
	const want = `package p

import "golang.org/x/net/http2/hpack"

var _ = hpack.HuffmanDecode
`
	testConfig{
		gopathOnly: true,
		module: packagestest.Module{
			Name: "foo.com",
			Files: fm{
				"vendor/golang.org/x/net/http2/hpack/huffman.go": "package hpack\nfunc HuffmanDecode() { }\n",
				"bar/x.go": input,
			},
		},
	}.processTest(t, "foo.com", "bar/x.go", nil, nil, want)
}

func TestFindStdlib(t *testing.T) {
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
		input := "package p\n"
		for _, sym := range tt.symbols {
			input += fmt.Sprintf("var _ = %s.%s\n", tt.pkg, sym)
		}
		testConfig{
			module: packagestest.Module{
				Name:  "foo.com",
				Files: fm{"x.go": input},
			},
		}.test(t, func(t *goimportTest) {
			buf, err := t.process("foo.com", "x.go", nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if got := string(buf); !strings.Contains(got, tt.want) {
				t.Errorf("Process(%q) = %q, wanted it to contain %q", input, buf, tt.want)
			}
		})
	}
}

// https://golang.org/issue/31814
func TestStdlibNotPrefixed(t *testing.T) {
	const input = `package p
var _ = bytes.Buffer
`
	const want = `package p

import "bytes"

var _ = bytes.Buffer
`
	// Force a scan of the stdlib.
	savedStdlib := stdlib
	defer func() { stdlib = savedStdlib }()
	stdlib = map[string][]string{}

	testConfig{
		module: packagestest.Module{
			Name: "ignored.com",
		},
	}.test(t, func(t *goimportTest) {
		// Run in GOROOT/src so that the std module shows up in go list -m all.
		t.env.WorkingDir = filepath.Join(t.env.GOROOT, "src")
		got, err := t.processNonModule(filepath.Join(t.env.GOROOT, "src/x.go"), []byte(input), nil)
		if err != nil {
			t.Fatalf("Process() = %v", err)
		}
		if string(got) != want {
			t.Errorf("Got:\n%s\nWant:\n%s", got, want)
		}
	})
}

type testConfig struct {
	gopathOnly bool
	module     packagestest.Module
	modules    []packagestest.Module
}

// fm is the type for a packagestest.Module's Files, abbreviated for shorter lines.
type fm map[string]interface{}

func (c testConfig) test(t *testing.T, fn func(*goimportTest)) {
	t.Helper()

	if c.module.Name != "" {
		c.modules = []packagestest.Module{c.module}
	}

	for _, exporter := range packagestest.All {
		t.Run(exporter.Name(), func(t *testing.T) {
			t.Helper()
			if c.gopathOnly && exporter.Name() == "Modules" {
				t.Skip("test marked GOPATH-only")
			}
			exported := packagestest.Export(t, exporter, c.modules)
			defer exported.Cleanup()

			env := make(map[string]string)
			for _, kv := range exported.Config.Env {
				split := strings.Split(kv, "=")
				k, v := split[0], split[1]
				env[k] = v
			}

			it := &goimportTest{
				T: t,
				env: &ProcessEnv{
					GOROOT:      env["GOROOT"],
					GOPATH:      env["GOPATH"],
					GO111MODULE: env["GO111MODULE"],
					GOSUMDB:     env["GOSUMDB"],
					WorkingDir:  exported.Config.Dir,
					Debug:       *testDebug,
					Logf:        log.Printf,
				},
				exported: exported,
			}
			if it.env.GOROOT == "" {
				// packagestest clears out GOROOT to work around https://golang.org/issue/32849,
				// which isn't relevant here. Fill it back in so we can find the standard library.
				it.env.GOROOT = build.Default.GOROOT
			}
			fn(it)
		})
	}
}

func (c testConfig) processTest(t *testing.T, module, file string, contents []byte, opts *Options, want string) {
	t.Helper()
	c.test(t, func(t *goimportTest) {
		t.Helper()
		t.assertProcessEquals(module, file, contents, opts, want)
	})
}

type goimportTest struct {
	*testing.T
	env      *ProcessEnv
	exported *packagestest.Exported
}

func (t *goimportTest) process(module, file string, contents []byte, opts *Options) ([]byte, error) {
	t.Helper()
	f := t.exported.File(module, file)
	if f == "" {
		t.Fatalf("%v not found in exported files (typo in filename?)", file)
	}
	return t.processNonModule(f, contents, opts)
}

func (t *goimportTest) processNonModule(file string, contents []byte, opts *Options) ([]byte, error) {
	if opts == nil {
		opts = &Options{Comments: true, TabIndent: true, TabWidth: 8}
	}
	// ProcessEnv is not safe for concurrent use. Make a copy.
	opts.Env = t.env.CopyConfig()
	return Process(file, contents, opts)
}

func (t *goimportTest) assertProcessEquals(module, file string, contents []byte, opts *Options, want string) {
	buf, err := t.process(module, file, contents, opts)
	if err != nil {
		t.Fatalf("Process() = %v", err)
	}
	if string(buf) != want {
		t.Errorf("Got:\n%s\nWant:\n%s", buf, want)
	}
}

// Tests that added imports are renamed when the import path's base doesn't
// match its package name.
func TestRenameWhenPackageNameMismatch(t *testing.T) {
	const input = `package main
 const Y = bar.X`

	const want = `package main

import bar "foo.com/foo/bar/baz"

const Y = bar.X
`
	testConfig{
		module: packagestest.Module{
			Name: "foo.com",
			Files: fm{
				"foo/bar/baz/x.go": "package bar \n const X = 1",
				"test/t.go":        input,
			},
		},
	}.processTest(t, "foo.com", "test/t.go", nil, nil, want)
}

// Tests that an existing import with badly mismatched path/name has its name
// correctly added. See #28645 and #29041.
func TestAddNameToMismatchedImport(t *testing.T) {
	const input = `package main

import (
"foo.com/a.thing"
"foo.com/surprise"
"foo.com/v1"
"foo.com/other/v2"
"foo.com/other/v3"
"foo.com/go-thing"
"foo.com/go-wrong"
)

var _ = []interface{}{bar.X, v1.Y, a.A, v2.V2, other.V3, thing.Thing, gow.Wrong}`

	const want = `package main

import (
	"foo.com/a.thing"
	"foo.com/go-thing"
	gow "foo.com/go-wrong"
	v2 "foo.com/other/v2"
	"foo.com/other/v3"
	bar "foo.com/surprise"
	v1 "foo.com/v1"
)

var _ = []interface{}{bar.X, v1.Y, a.A, v2.V2, other.V3, thing.Thing, gow.Wrong}
`

	testConfig{
		module: packagestest.Module{
			Name: "foo.com",
			Files: fm{
				"a.thing/a.go":  "package a \n const A = 1",
				"surprise/x.go": "package bar \n const X = 1",
				"v1/x.go":       "package v1 \n const Y = 1",
				"other/v2/y.go": "package v2 \n const V2 = 1",
				"other/v3/z.go": "package other \n const V3 = 1",
				"go-thing/b.go": "package thing \n const Thing = 1",
				"go-wrong/b.go": "package gow \n const Wrong = 1",
				"test/t.go":     input,
			},
		},
	}.processTest(t, "foo.com", "test/t.go", nil, nil, want)
}

// Tests that the LocalPrefix option causes imports
// to be added into a later group (num=3).
func TestLocalPrefix(t *testing.T) {
	tests := []struct {
		name        string
		modules     []packagestest.Module
		localPrefix string
		src         string
		want        string
	}{
		{
			name: "one_local",
			modules: []packagestest.Module{
				{
					Name: "foo.com",
					Files: fm{
						"bar/bar.go": "package bar \n const X = 1",
					},
				},
			},
			localPrefix: "foo.com/",
			src:         "package main \n const Y = bar.X \n const _ = runtime.GOOS",
			want: `package main

import (
	"runtime"

	"foo.com/bar"
)

const Y = bar.X
const _ = runtime.GOOS
`,
		},
		{
			name: "two_local",
			modules: []packagestest.Module{
				{
					Name: "foo.com",
					Files: fm{
						"foo/foo.go":     "package foo \n const X = 1",
						"foo/bar/bar.go": "package bar \n const X = 1",
					},
				},
			},
			localPrefix: "foo.com/foo",
			src:         "package main \n const Y = bar.X \n const Z = foo.X \n const _ = runtime.GOOS",
			want: `package main

import (
	"runtime"

	"foo.com/foo"
	"foo.com/foo/bar"
)

const Y = bar.X
const Z = foo.X
const _ = runtime.GOOS
`,
		},
		{
			name: "three_prefixes",
			modules: []packagestest.Module{
				{
					Name:  "example.org/pkg",
					Files: fm{"pkg.go": "package pkg \n const A = 1"},
				},
				{
					Name:  "foo.com",
					Files: fm{"bar/bar.go": "package bar \n const B = 1"},
				},
				{
					Name:  "code.org/r/p",
					Files: fm{"expproj/expproj.go": "package expproj \n const C = 1"},
				},
			},
			localPrefix: "example.org/pkg,foo.com/,code.org",
			src:         "package main \n const X = pkg.A \n const Y = bar.B \n const Z = expproj.C \n const _ = runtime.GOOS",
			want: `package main

import (
	"runtime"

	"code.org/r/p/expproj"
	"example.org/pkg"
	"foo.com/bar"
)

const X = pkg.A
const Y = bar.B
const Z = expproj.C
const _ = runtime.GOOS
`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testConfig{
				// The module being processed has to be first so it's the primary module.
				modules: append([]packagestest.Module{{
					Name:  "test.com",
					Files: fm{"t.go": tt.src},
				}}, tt.modules...),
			}.test(t, func(t *goimportTest) {
				t.env.LocalPrefix = tt.localPrefix
				t.assertProcessEquals("test.com", "t.go", nil, nil, tt.want)
			})
		})
	}
}

// Tests that "package documentation" files are ignored.
func TestIgnoreDocumentationPackage(t *testing.T) {
	const input = `package x

const Y = foo.X
`
	const want = `package x

import "foo.com/foo"

const Y = foo.X
`

	testConfig{
		module: packagestest.Module{
			Name: "foo.com",
			Files: fm{
				"foo/foo.go": "package foo\nconst X = 1\n",
				"foo/doc.go": "package documentation \n // just to confuse things\n",
				"x/x.go":     input,
			},
		},
	}.processTest(t, "foo.com", "x/x.go", nil, nil, want)
}

// Tests importPathToNameGoPathParse and in particular that it stops
// after finding the first non-documentation package name, not
// reporting an error on inconsistent package names (since it should
// never make it that far).
func TestImportPathToNameGoPathParse(t *testing.T) {
	testConfig{
		module: packagestest.Module{
			Name: "example.net/pkg",
			Files: fm{
				"doc.go": "package documentation\n", // ignored
				"gen.go": "package main\n",          // also ignored
				"pkg.go": "package the_pkg_name_to_find\n  and this syntax error is ignored because of parser.PackageClauseOnly",
				"z.go":   "package inconsistent\n", // inconsistent but ignored
			},
		},
	}.test(t, func(t *goimportTest) {
		if strings.Contains(t.Name(), "GoPackages") {
			t.Skip("go/packages does not ignore package main")
		}
		r := t.env.GetResolver()
		srcDir := filepath.Dir(t.exported.File("example.net/pkg", "z.go"))
		names, err := r.loadPackageNames([]string{"example.net/pkg"}, srcDir)
		if err != nil {
			t.Fatal(err)
		}
		const want = "the_pkg_name_to_find"
		if got := names["example.net/pkg"]; got != want {
			t.Errorf("loadPackageNames(..) = %q; want %q", got, want)
		}
	})
}

func TestIgnoreConfiguration(t *testing.T) {
	const input = `package x

const _ = pkg.X
`
	const want = `package x

import "foo.com/otherwise-longer-so-worse-example/foo/pkg"

const _ = pkg.X
`

	testConfig{
		gopathOnly: true,
		module: packagestest.Module{
			Name: "foo.com",
			Files: fm{
				"../.goimportsignore":                              "# comment line\n\n foo.com/example", // tests comment, blank line, whitespace trimming
				"example/pkg/pkg.go":                               "package pkg\nconst X = 1",
				"otherwise-longer-so-worse-example/foo/pkg/pkg.go": "package pkg\nconst X = 1",
				"x/x.go": input,
			},
		},
	}.processTest(t, "foo.com", "x/x.go", nil, nil, want)
}

// Skip "node_modules" directory.
func TestSkipNodeModules(t *testing.T) {
	const input = `package x

const _ = pkg.X
`
	const want = `package x

import "foo.com/otherwise-longer/not_modules/pkg"

const _ = pkg.X
`

	testConfig{
		gopathOnly: true,
		module: packagestest.Module{
			Name: "foo.com",
			Files: fm{
				"example/node_modules/pkg/a.go":         "package pkg\nconst X = 1",
				"otherwise-longer/not_modules/pkg/a.go": "package pkg\nconst X = 1",
				"x/x.go":                                input,
			},
		},
	}.processTest(t, "foo.com", "x/x.go", nil, nil, want)
}

// Tests that package global variables with the same name and function name as
// a function in a separate package do not result in an import which masks
// the global variable
func TestGlobalImports(t *testing.T) {
	const usesGlobal = `package pkg

func doSomething() {
	t := time.Now()
}
`

	const declaresGlobal = `package pkg

type Time struct{}

func (t Time) Now() Time {
	return Time{}
}

var time Time
`

	testConfig{
		module: packagestest.Module{
			Name: "foo.com",
			Files: fm{
				"pkg/uses.go":   usesGlobal,
				"pkg/global.go": declaresGlobal,
			},
		},
	}.processTest(t, "foo.com", "pkg/uses.go", nil, nil, usesGlobal)
}

// Some people put multiple packages' files in the same directory. Globals
// declared in other packages should be ignored.
func TestGlobalImports_DifferentPackage(t *testing.T) {
	const declaresGlobal = `package main
var fmt int
`
	const input = `package pkg
var _ = fmt.Printf
`
	const want = `package pkg

import "fmt"

var _ = fmt.Printf
`

	testConfig{
		module: packagestest.Module{
			Name: "foo.com",
			Files: fm{
				"pkg/main.go": declaresGlobal,
				"pkg/uses.go": input,
			},
		},
	}.processTest(t, "foo.com", "pkg/uses.go", nil, nil, want)
}

func TestGlobalImports_MultipleMains(t *testing.T) {
	const declaresGlobal = `package main
var fmt int
`
	const input = `package main
import "fmt"
var _, _ = fmt.Printf, bytes.Equal
`
	const want = `package main

import (
	"bytes"
	"fmt"
)

var _, _ = fmt.Printf, bytes.Equal
`

	testConfig{
		module: packagestest.Module{
			Name: "foo.com",
			Files: fm{
				"pkg/main.go": declaresGlobal,
				"pkg/uses.go": input,
			},
		},
	}.processTest(t, "foo.com", "pkg/uses.go", nil, nil, want)
}

// Tests that sibling files - other files in the same package - can provide an
// import that may not be the default one otherwise.
func TestSiblingImports(t *testing.T) {

	// provide is the sibling file that provides the desired import.
	const provide = `package siblingimporttest

import "local/log"
import "my/bytes"
import renamed "fmt"

func LogSomething() {
	log.Print("Something")
	bytes.SomeFunc()
	renamed.Println("Something")
}
`

	// need is the file being tested that needs the import.
	const need = `package siblingimporttest

var _ = bytes.Buffer{}

func LogSomethingElse() {
	log.Print("Something else")
	renamed.Println("Yet another")
}
`

	// want is the expected result file
	const want = `package siblingimporttest

import (
	"bytes"
	renamed "fmt"
	"local/log"
)

var _ = bytes.Buffer{}

func LogSomethingElse() {
	log.Print("Something else")
	renamed.Println("Yet another")
}
`

	testConfig{
		module: packagestest.Module{
			Name: "foo.com",
			Files: fm{
				"p/needs_import.go":    need,
				"p/provides_import.go": provide,
			},
		},
	}.processTest(t, "foo.com", "p/needs_import.go", nil, nil, want)
}

// Tests #29180: a sibling import of the right package with the wrong name is used.
func TestSiblingImport_Misnamed(t *testing.T) {
	const sibling = `package main
import renamed "fmt"
var _ = renamed.Printf
`
	const input = `package pkg
var _ = fmt.Printf
`
	const want = `package pkg

import "fmt"

var _ = fmt.Printf
`

	testConfig{
		module: packagestest.Module{
			Name: "foo.com",
			Files: fm{
				"pkg/main.go": sibling,
				"pkg/uses.go": input,
			},
		},
	}.processTest(t, "foo.com", "pkg/uses.go", nil, nil, want)

}

// Tests that an input file's own package is ignored.
func TestIgnoreOwnPackage(t *testing.T) {
	const input = `package pkg

const _ = pkg.X
`
	const want = `package pkg

const _ = pkg.X
`

	testConfig{
		module: packagestest.Module{
			Name: "foo.com",
			Files: fm{
				"pkg/a.go": "package pkg\nconst X = 1",
				"pkg/b.go": input,
			},
		},
	}.processTest(t, "foo.com", "pkg/b.go", nil, nil, want)
}

func TestExternalTestImportsPackageUnderTest(t *testing.T) {
	const provide = `package pkg
func DoIt(){}
`
	const input = `package pkg_test

var _ = pkg.DoIt`

	const want = `package pkg_test

import "foo.com/pkg"

var _ = pkg.DoIt
`

	testConfig{
		module: packagestest.Module{
			Name: "foo.com",
			Files: fm{
				"pkg/provide.go": provide,
				"pkg/x_test.go":  input,
			},
		},
	}.processTest(t, "foo.com", "pkg/x_test.go", nil, nil, want)
}

func TestPkgIsCandidate(t *testing.T) {
	tests := []struct {
		name     string
		filename string
		pkgIdent string
		pkg      *pkg
		want     bool
	}{
		{
			name:     "normal_match",
			filename: "/gopath/src/my/pkg/pkg.go",
			pkgIdent: "client",
			pkg: &pkg{
				dir:             "/gopath/src/client",
				importPathShort: "client",
			},
			want: true,
		},
		{
			name:     "no_match",
			filename: "/gopath/src/my/pkg/pkg.go",
			pkgIdent: "zzz",
			pkg: &pkg{
				dir:             "/gopath/src/client",
				importPathShort: "client",
			},
			want: false,
		},
		{
			name:     "match_too_early",
			filename: "/gopath/src/my/pkg/pkg.go",
			pkgIdent: "client",
			pkg: &pkg{
				dir:             "/gopath/src/client/foo/foo/foo",
				importPathShort: "client/foo/foo",
			},
			want: false,
		},
		{
			name:     "substring_match",
			filename: "/gopath/src/my/pkg/pkg.go",
			pkgIdent: "client",
			pkg: &pkg{
				dir:             "/gopath/src/foo/go-client",
				importPathShort: "foo/go-client",
			},
			want: true,
		},
		{
			name:     "hidden_internal",
			filename: "/gopath/src/my/pkg/pkg.go",
			pkgIdent: "client",
			pkg: &pkg{
				dir:             "/gopath/src/foo/internal/client",
				importPathShort: "foo/internal/client",
			},
			want: false,
		},
		{
			name:     "visible_internal",
			filename: "/gopath/src/foo/bar.go",
			pkgIdent: "client",
			pkg: &pkg{
				dir:             "/gopath/src/foo/internal/client",
				importPathShort: "foo/internal/client",
			},
			want: true,
		},
		{
			name:     "invisible_vendor",
			filename: "/gopath/src/foo/bar.go",
			pkgIdent: "client",
			pkg: &pkg{
				dir:             "/gopath/src/other/vendor/client",
				importPathShort: "client",
			},
			want: false,
		},
		{
			name:     "visible_vendor",
			filename: "/gopath/src/foo/bar.go",
			pkgIdent: "client",
			pkg: &pkg{
				dir:             "/gopath/src/foo/vendor/client",
				importPathShort: "client",
			},
			want: true,
		},
		{
			name:     "match_with_hyphens",
			filename: "/gopath/src/foo/bar.go",
			pkgIdent: "socketio",
			pkg: &pkg{
				dir:             "/gopath/src/foo/socket-io",
				importPathShort: "foo/socket-io",
			},
			want: true,
		},
		{
			name:     "match_with_mixed_case",
			filename: "/gopath/src/foo/bar.go",
			pkgIdent: "fooprod",
			pkg: &pkg{
				dir:             "/gopath/src/foo/FooPROD",
				importPathShort: "foo/FooPROD",
			},
			want: true,
		},
		{
			name:     "matches_with_hyphen_and_caps",
			filename: "/gopath/src/foo/bar.go",
			pkgIdent: "fooprod",
			pkg: &pkg{
				dir:             "/gopath/src/foo/Foo-PROD",
				importPathShort: "foo/Foo-PROD",
			},
			want: true,
		},
	}
	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			refs := references{tt.pkgIdent: nil}
			got := pkgIsCandidate(tt.filename, refs, tt.pkg)
			if got != tt.want {
				t.Errorf("test %d. pkgIsCandidate(%q, %q, %+v) = %v; want %v",
					i, tt.filename, tt.pkgIdent, *tt.pkg, got, tt.want)
			}
		})
	}
}

// Issue 20941: this used to panic on Windows.
func TestProcessStdin(t *testing.T) {
	testConfig{
		module: packagestest.Module{
			Name: "foo.com",
		},
	}.test(t, func(t *goimportTest) {
		got, err := t.processNonModule("<standard input>", []byte("package main\nfunc main() {\n\tfmt.Println(123)\n}\n"), nil)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(got), `"fmt"`) {
			t.Errorf("expected fmt import; got: %s", got)
		}
	})
}

// Tests LocalPackagePromotion when there is a local package that matches, it
// should be the closest match.
// https://golang.org/issues/17557
func TestLocalPackagePromotion(t *testing.T) {
	const input = `package main
var c = &config.SystemConfig{}
`
	const want = `package main

import "mycompany.net/tool/config"

var c = &config.SystemConfig{}
`

	testConfig{
		modules: []packagestest.Module{
			{
				Name:  "config.net/config",
				Files: fm{"config.go": "package config\n type SystemConfig struct {}"}, // Will match but should not be first choice
			},
			{
				Name:  "mycompany.net/config",
				Files: fm{"config.go": "package config\n type SystemConfig struct {}"}, // Will match but should not be first choice
			},
			{
				Name: "mycompany.net/tool",
				Files: fm{
					"config/config.go": "package config\n type SystemConfig struct {}", // Local package should be promoted over shorter package
					"main.go":          input,
				},
			},
		},
	}.processTest(t, "mycompany.net/tool", "main.go", nil, nil, want)
}

// Tests FindImportInLocalGoFiles looks at the import lines for other Go files in the
// local directory, since the user is likely to import the same packages in the current
// Go file.  If an import is found that satisfies the need, it should be used over the
// standard library.
// https://golang.org/issues/17557
func TestFindImportInLocalGoFiles(t *testing.T) {
	const input = `package main
 var _ = &bytes.Buffer{}`

	const want = `package main

import "bytes.net/bytes"

var _ = &bytes.Buffer{}
`
	testConfig{
		modules: []packagestest.Module{
			{
				Name: "mycompany.net/tool",
				Files: fm{
					"io.go":   "package main\n import \"bytes.net/bytes\"\n var _ = &bytes.Buffer{}", // Contains package import that will cause stdlib to be ignored
					"main.go": input,
				},
			},
			{
				Name:  "bytes.net/bytes",
				Files: fm{"bytes.go": "package bytes\n type Buffer struct {}"}, // Should be selected over standard library
			},
		},
	}.processTest(t, "mycompany.net/tool", "main.go", nil, nil, want)
}

func TestInMemoryFile(t *testing.T) {
	const input = `package main
 var _ = &bytes.Buffer{}`

	const want = `package main

import "bytes"

var _ = &bytes.Buffer{}
`
	testConfig{
		module: packagestest.Module{
			Name:  "foo.com",
			Files: fm{"x.go": "package x\n"},
		},
	}.processTest(t, "foo.com", "x.go", []byte(input), nil, want)
}

func TestImportNoGoFiles(t *testing.T) {
	const input = `package main
 var _ = &bytes.Buffer{}`

	const want = `package main

import "bytes"

var _ = &bytes.Buffer{}
`
	testConfig{
		module: packagestest.Module{
			Name: "mycompany.net",
		},
	}.test(t, func(t *goimportTest) {
		buf, err := t.processNonModule("mycompany.net/tool/main.go", []byte(input), nil)
		if err != nil {
			t.Fatalf("Process() = %v", err)
		}
		if string(buf) != want {
			t.Errorf("Got:\n%s\nWant:\n%s", buf, want)
		}
	})

}

// Ensures a token as large as 500000 bytes can be handled
// https://golang.org/issues/18201
func TestProcessLargeToken(t *testing.T) {
	largeString := strings.Repeat("x", 500000)

	input := `package testimports

import (
	"bytes"
)

const s = fmt.Sprintf("%s", "` + largeString + `")
var _ = bytes.Buffer{}

// end
`

	want := `package testimports

import (
	"bytes"
	"fmt"
)

const s = fmt.Sprintf("%s", "` + largeString + `")

var _ = bytes.Buffer{}

// end
`

	testConfig{
		module: packagestest.Module{
			Name:  "foo.com",
			Files: fm{"foo.go": input},
		},
	}.processTest(t, "foo.com", "foo.go", nil, nil, want)
}

// Tests that an external test package will import the package under test if it
// also uses symbols exported only in test files.
// https://golang.org/issues/29979
func TestExternalTest(t *testing.T) {
	const input = `package a_test
func TestX() {
	a.X()
	a.Y()
}
`
	const want = `package a_test

import "foo.com/a"

func TestX() {
	a.X()
	a.Y()
}
`

	testConfig{
		modules: []packagestest.Module{
			{
				Name: "foo.com/a",
				Files: fm{
					"a.go":           "package a\n func X() {}",
					"export_test.go": "package a\n func Y() {}",
					"a_test.go":      input,
				},
			},
		},
	}.processTest(t, "foo.com/a", "a_test.go", nil, nil, want)
}

// TestStdLibGetCandidates tests that get packages finds std library packages
// with correct priorities.
func TestGetCandidates(t *testing.T) {
	type res struct {
		relevance  int
		name, path string
	}
	want := []res{
		{0, "bytes", "bytes"},
		{0, "http", "net/http"},
		{0, "rand", "crypto/rand"},
		{0, "bar", "bar.com/bar"},
		{0, "foo", "foo.com/foo"},
	}

	testConfig{
		modules: []packagestest.Module{
			{
				Name:  "bar.com",
				Files: fm{"bar/bar.go": "package bar\n"},
			},
			{
				Name:  "foo.com",
				Files: fm{"foo/foo.go": "package foo\n"},
			},
		},
	}.test(t, func(t *goimportTest) {
		var mu sync.Mutex
		var got []res
		add := func(c ImportFix) {
			mu.Lock()
			defer mu.Unlock()
			for _, w := range want {
				if c.StmtInfo.ImportPath == w.path {
					got = append(got, res{c.Relevance, c.IdentName, c.StmtInfo.ImportPath})
				}
			}
		}
		if err := getAllCandidates(context.Background(), add, "", "x.go", "x", t.env); err != nil {
			t.Fatalf("GetAllCandidates() = %v", err)
		}
		// Sort, then clear out relevance so it doesn't mess up the DeepEqual.
		sort.Slice(got, func(i, j int) bool {
			ri, rj := got[i], got[j]
			if ri.relevance != rj.relevance {
				return ri.relevance > rj.relevance // Highest first.
			}
			return ri.name < rj.name
		})
		for i := range got {
			got[i].relevance = 0
		}
		if !reflect.DeepEqual(want, got) {
			t.Errorf("wanted stdlib results in order %v, got %v", want, got)
		}
	})
}

func TestGetPackageCompletions(t *testing.T) {
	type res struct {
		relevance          int
		name, path, symbol string
	}
	want := []res{
		{0, "rand", "math/rand", "Seed"},
		{0, "rand", "bar.com/rand", "Bar"},
	}

	testConfig{
		modules: []packagestest.Module{
			{
				Name:  "bar.com",
				Files: fm{"rand/bar.go": "package rand\nvar Bar int\n"},
			},
		},
	}.test(t, func(t *goimportTest) {
		var mu sync.Mutex
		var got []res
		add := func(c PackageExport) {
			mu.Lock()
			defer mu.Unlock()
			for _, csym := range c.Exports {
				for _, w := range want {
					if c.Fix.StmtInfo.ImportPath == w.path && csym == w.symbol {
						got = append(got, res{c.Fix.Relevance, c.Fix.IdentName, c.Fix.StmtInfo.ImportPath, csym})
					}
				}
			}
		}
		if err := getPackageExports(context.Background(), add, "rand", "x.go", "x", t.env); err != nil {
			t.Fatalf("getPackageCompletions() = %v", err)
		}
		// Sort, then clear out relevance so it doesn't mess up the DeepEqual.
		sort.Slice(got, func(i, j int) bool {
			ri, rj := got[i], got[j]
			if ri.relevance != rj.relevance {
				return ri.relevance > rj.relevance // Highest first.
			}
			return ri.name < rj.name
		})
		for i := range got {
			got[i].relevance = 0
		}
		if !reflect.DeepEqual(want, got) {
			t.Errorf("wanted stdlib results in order %v, got %v", want, got)
		}
	})
}

// Tests #34895: process should not panic on concurrent calls.
func TestConcurrentProcess(t *testing.T) {
	testConfig{
		module: packagestest.Module{
			Name: "foo.com",
			Files: fm{
				"p/first.go": `package foo

func _() {
	fmt.Println()
}
`,
				"p/second.go": `package foo

import "fmt"

func _() {
	fmt.Println()
	imports.Bar() // not imported.
}
`,
			},
		},
	}.test(t, func(t *goimportTest) {
		var (
			n  = 10
			wg sync.WaitGroup
		)
		wg.Add(n)
		for i := 0; i < n; i++ {
			go func() {
				defer wg.Done()
				_, err := t.process("foo.com", "p/first.go", nil, nil)
				if err != nil {
					t.Fatal(err)
				}
			}()
		}
		wg.Wait()
	})
}
