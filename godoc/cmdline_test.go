// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

import (
	"bytes"
	"go/build"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"testing"
	"text/template"

	"golang.org/x/tools/godoc/vfs"
	"golang.org/x/tools/godoc/vfs/mapfs"
)

// setupGoroot creates temporary directory to act as GOROOT when running tests
// that depend upon the build package.  It updates build.Default to point to the
// new GOROOT.
// It returns a function that can be called to reset build.Default and remove
// the temporary directory.
func setupGoroot(t *testing.T) (cleanup func()) {
	var stdLib = map[string]string{
		"src/fmt/fmt.go": `// Package fmt implements formatted I/O.
package fmt

type Stringer interface {
	String() string
}
`,
	}
	goroot, err := ioutil.TempDir("", "cmdline_test")
	if err != nil {
		t.Fatal(err)
	}
	origContext := build.Default
	build.Default = build.Context{
		GOROOT:   goroot,
		Compiler: "gc",
	}
	for relname, contents := range stdLib {
		name := filepath.Join(goroot, relname)
		if err := os.MkdirAll(filepath.Dir(name), 0770); err != nil {
			t.Fatal(err)
		}
		if err := ioutil.WriteFile(name, []byte(contents), 0770); err != nil {
			t.Fatal(err)
		}
	}

	return func() {
		if err := os.RemoveAll(goroot); err != nil {
			t.Log(err)
		}
		build.Default = origContext
	}
}

func TestPaths(t *testing.T) {
	cleanup := setupGoroot(t)
	defer cleanup()

	pres := &Presentation{
		pkgHandler: handlerServer{
			fsRoot: "/fsroot",
		},
	}
	fs := make(vfs.NameSpace)

	absPath := "/foo/fmt"
	if runtime.GOOS == "windows" {
		absPath = `c:\foo\fmt`
	}

	for _, tc := range []struct {
		desc   string
		path   string
		expAbs string
		expRel string
	}{
		{
			"Absolute path",
			absPath,
			"/target",
			"/target",
		},
		{
			"Local import",
			"../foo/fmt",
			"/target",
			"/target",
		},
		{
			"Import",
			"fmt",
			"/target",
			"fmt",
		},
		{
			"Default",
			"unknownpkg",
			"/fsroot/unknownpkg",
			"unknownpkg",
		},
	} {
		abs, rel := paths(fs, pres, tc.path)
		if abs != tc.expAbs || rel != tc.expRel {
			t.Errorf("%s: paths(%q) = %s,%s; want %s,%s", tc.desc, tc.path, abs, rel, tc.expAbs, tc.expRel)
		}
	}
}

func TestMakeRx(t *testing.T) {
	for _, tc := range []struct {
		desc  string
		names []string
		exp   string
	}{
		{
			desc:  "empty string",
			names: []string{""},
			exp:   `^$`,
		},
		{
			desc:  "simple text",
			names: []string{"a"},
			exp:   `^a$`,
		},
		{
			desc:  "two words",
			names: []string{"foo", "bar"},
			exp:   `^foo$|^bar$`,
		},
		{
			desc:  "word & non-trivial",
			names: []string{"foo", `ab?c`},
			exp:   `^foo$|ab?c`,
		},
		{
			desc:  "bad regexp",
			names: []string{`(."`},
			exp:   `(."`,
		},
	} {
		expRE, expErr := regexp.Compile(tc.exp)
		if re, err := makeRx(tc.names); !reflect.DeepEqual(err, expErr) && !reflect.DeepEqual(re, expRE) {
			t.Errorf("%s: makeRx(%v) = %q,%q; want %q,%q", tc.desc, tc.names, re, err, expRE, expErr)
		}
	}
}

func TestCommandLine(t *testing.T) {
	cleanup := setupGoroot(t)
	defer cleanup()
	mfs := mapfs.New(map[string]string{
		"src/bar/bar.go": `// Package bar is an example.
package bar
`,
		"src/foo/foo.go": `// Package foo.
package foo

// First function is first.
func First() {
}

// Second function is second.
func Second() {
}

// unexported function is third.
func unexported() {
}
`,
		"src/gen/gen.go": `// Package gen
package gen

//line notgen.go:3
// F doc //line 1 should appear
// line 2 should appear
func F()
//line foo.go:100`, // no newline on end to check corner cases!
		"src/vet/vet.go": `// Package vet
package vet
`,
		"src/cmd/go/doc.go": `// The go command
package main
`,
		"src/cmd/gofmt/doc.go": `// The gofmt command
package main
`,
		"src/cmd/vet/vet.go": `// The vet command
package main
`,
	})
	fs := make(vfs.NameSpace)
	fs.Bind("/", mfs, "/", vfs.BindReplace)
	c := NewCorpus(fs)
	p := &Presentation{Corpus: c}
	p.cmdHandler = handlerServer{
		p:       p,
		c:       c,
		pattern: "/cmd/",
		fsRoot:  "/src",
	}
	p.pkgHandler = handlerServer{
		p:       p,
		c:       c,
		pattern: "/pkg/",
		fsRoot:  "/src",
		exclude: []string{"/src/cmd"},
	}
	p.initFuncMap()
	p.PackageText = template.Must(template.New("PackageText").Funcs(p.FuncMap()).Parse(`{{$info := .}}{{$filtered := .IsFiltered}}{{if $filtered}}{{range .PAst}}{{range .Decls}}{{node $info .}}{{end}}{{end}}{{else}}{{with .PAst}}{{range $filename, $ast := .}}{{$filename}}:
{{node $ $ast}}{{end}}{{end}}{{end}}{{with .PDoc}}{{if $.IsMain}}COMMAND {{.Doc}}{{else}}PACKAGE {{.Doc}}{{end}}{{with .Funcs}}
{{range .}}{{node $ .Decl}}
{{comment_text .Doc "    " "\t"}}{{end}}{{end}}{{end}}`))

	for _, tc := range []struct {
		desc string
		args []string
		all  bool
		exp  string
		err  bool
	}{
		{
			desc: "standard package",
			args: []string{"fmt"},
			exp:  "PACKAGE Package fmt implements formatted I/O.\n",
		},
		{
			desc: "package",
			args: []string{"bar"},
			exp:  "PACKAGE Package bar is an example.\n",
		},
		{
			desc: "package w. filter",
			args: []string{"foo", "First"},
			exp:  "PACKAGE \nfunc First()\n    First function is first.\n",
		},
		{
			desc: "package w. bad filter",
			args: []string{"foo", "DNE"},
			exp:  "PACKAGE ",
		},
		{
			desc: "source mode",
			args: []string{"src/bar"},
			exp:  "bar/bar.go:\n// Package bar is an example.\npackage bar\n",
		},
		{
			desc: "source mode w. filter",
			args: []string{"src/foo", "Second"},
			exp:  "// Second function is second.\nfunc Second() {\n}",
		},
		{
			desc: "package w. unexported filter",
			args: []string{"foo", "unexported"},
			all:  true,
			exp:  "PACKAGE \nfunc unexported()\n    unexported function is third.\n",
		},
		{
			desc: "package w. unexported filter",
			args: []string{"foo", "unexported"},
			all:  false,
			exp:  "PACKAGE ",
		},
		{
			desc: "package w. //line comments",
			args: []string{"gen", "F"},
			exp:  "PACKAGE \nfunc F()\n    F doc //line 1 should appear line 2 should appear\n",
		},
		{
			desc: "command",
			args: []string{"go"},
			exp:  "COMMAND The go command\n",
		},
		{
			desc: "forced command",
			args: []string{"cmd/gofmt"},
			exp:  "COMMAND The gofmt command\n",
		},
		{
			desc: "bad arg",
			args: []string{"doesnotexist"},
			err:  true,
		},
		{
			desc: "both command and package",
			args: []string{"vet"},
			exp:  "use 'godoc cmd/vet' for documentation on the vet command \n\nPACKAGE Package vet\n",
		},
		{
			desc: "root directory",
			args: []string{"/"},
			exp:  "",
		},
	} {
		p.AllMode = tc.all
		w := new(bytes.Buffer)
		err := CommandLine(w, fs, p, tc.args)
		if got, want := w.String(), tc.exp; got != want || tc.err == (err == nil) {
			t.Errorf("%s: CommandLine(%v), All(%v) = %q (%v); want %q (%v)",
				tc.desc, tc.args, tc.all, got, err, want, tc.err)
		}
	}
}
