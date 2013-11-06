package astutil

import (
	"bytes"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"

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
	name string
	pkg  string
	in   string
	out  string
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
}

func TestAddImport(t *testing.T) {
	for _, test := range addTests {
		file := parse(t, test.name, test.in)
		AddImport(file, test.pkg)
		if got := print(t, test.name, file); got != test.out {
			t.Errorf("%s:\ngot: %s\nwant: %s", test.name, got, test.out)
		}
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
}

func TestDeleteImport(t *testing.T) {
	for _, test := range deleteTests {
		file := parse(t, test.name, test.in)
		DeleteImport(file, test.pkg)
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
		RewriteImport(file, test.srcPkg, test.dstPkg)
		if got := print(t, test.name, file); got != test.out {
			t.Errorf("%s:\ngot: %s\nwant: %s", test.name, got, test.out)
		}
	}
}
