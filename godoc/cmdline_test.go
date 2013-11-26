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

	"code.google.com/p/go.tools/godoc/vfs"
	"code.google.com/p/go.tools/godoc/vfs/mapfs"
)

// setupGoroot creates temporary directory to act as GOROOT when running tests
// that depend upon the build package.  It updates build.Default to point to the
// new GOROOT.
// It returns a function that can be called to reset build.Default and remove
// the temporary directory.
func setupGoroot(t *testing.T) (cleanup func()) {
	var stdLib = map[string]string{
		"src/pkg/fmt/fmt.go": `// Package fmt implements formatted I/O.
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
		"src/pkg/bar/bar.go": `// Package bar is an example.
package bar
`,
		"src/cmd/go/doc.go": `// The go command
package main
`,
		"src/cmd/gofmt/doc.go": `// The gofmt command
package main
`,
	})
	fs := make(vfs.NameSpace)
	fs.Bind("/", mfs, "/", vfs.BindReplace)
	c := NewCorpus(fs)
	p := &Presentation{Corpus: c}
	p.cmdHandler = handlerServer{p, c, "/cmd/", "/src/cmd"}
	p.pkgHandler = handlerServer{p, c, "/pkg/", "/src/pkg"}
	p.initFuncMap()
	p.PackageText = template.Must(template.New("PackageText").Funcs(p.FuncMap()).Parse(`{{with .PAst}}{{node $ .}}{{end}}{{with .PDoc}}{{if $.IsMain}}COMMAND {{.Doc}}{{else}}PACKAGE {{.Doc}}{{end}}{{end}}`))

	for _, tc := range []struct {
		desc string
		args []string
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
			desc: "source mode",
			args: []string{"src/bar"},
			exp:  "// Package bar is an example.\npackage bar\n",
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
	} {
		w := new(bytes.Buffer)
		err := CommandLine(w, fs, p, tc.args)
		if got, want := w.String(), tc.exp; got != want || tc.err == (err == nil) {
			t.Errorf("%s: CommandLine(%v) = %q,%v; want %q,%v",
				tc.desc, tc.args, got, err, want, tc.err)
		}
	}
}
