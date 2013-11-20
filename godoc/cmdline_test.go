package godoc

import (
	"bytes"
	"reflect"
	"regexp"
	"testing"
	"text/template"

	"code.google.com/p/go.tools/godoc/vfs"
	"code.google.com/p/go.tools/godoc/vfs/mapfs"
)

func TestPaths(t *testing.T) {
	pres := &Presentation{
		pkgHandler: handlerServer{
			fsRoot: "/fsroot",
		},
	}
	fs := make(vfs.NameSpace)

	for _, tc := range []struct {
		desc   string
		path   string
		expAbs string
		expRel string
	}{
		{
			"Absolute path",
			"/foo/fmt",
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
			args: []string{"runtime/race"},
			exp: `PACKAGE Package race implements data race detection logic.
No public interface is provided.
For details about the race detector see
http://golang.org/doc/articles/race_detector.html
`,
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
