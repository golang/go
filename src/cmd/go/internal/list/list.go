// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package list implements the ``go list'' command.
package list

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"os"
	"sort"
	"strings"
	"text/template"

	"cmd/go/internal/base"
	"cmd/go/internal/cache"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/work"
)

var CmdList = &base.Command{
	UsageLine: "list [-cgo] [-deps] [-e] [-export] [-f format] [-json] [-test] [build flags] [packages]",
	Short:     "list packages",
	Long: `
List lists the packages named by the import paths, one per line.

The default output shows the package import path:

    bytes
    encoding/json
    github.com/gorilla/mux
    golang.org/x/net/html

The -f flag specifies an alternate format for the list, using the
syntax of package template. The default output is equivalent to -f
'{{.ImportPath}}'. The struct being passed to the template is:

    type Package struct {
        Dir           string // directory containing package sources
        ImportPath    string // import path of package in dir
        ImportComment string // path in import comment on package statement
        Name          string // package name
        Doc           string // package documentation string
        Target        string // install path
        Shlib         string // the shared library that contains this package (only set when -linkshared)
        Goroot        bool   // is this package in the Go root?
        Standard      bool   // is this package part of the standard Go library?
        Stale         bool   // would 'go install' do anything for this package?
        StaleReason   string // explanation for Stale==true
        Root          string // Go root or Go path dir containing this package
        ConflictDir   string // this directory shadows Dir in $GOPATH
        BinaryOnly    bool   // binary-only package: cannot be recompiled from sources
        ForTest       string // package is only for use in named test
        DepOnly       bool   // package is only a dependency, not explicitly listed
        Export        string // file containing export data (when using -export)

        // Source files
        GoFiles        []string // .go source files (excluding CgoFiles, TestGoFiles, XTestGoFiles)
        CgoFiles       []string // .go sources files that import "C"
        IgnoredGoFiles []string // .go sources ignored due to build constraints
        CFiles         []string // .c source files
        CXXFiles       []string // .cc, .cxx and .cpp source files
        MFiles         []string // .m source files
        HFiles         []string // .h, .hh, .hpp and .hxx source files
        FFiles         []string // .f, .F, .for and .f90 Fortran source files
        SFiles         []string // .s source files
        SwigFiles      []string // .swig files
        SwigCXXFiles   []string // .swigcxx files
        SysoFiles      []string // .syso object files to add to archive
        TestGoFiles    []string // _test.go files in package
        XTestGoFiles   []string // _test.go files outside package

        // Cgo directives
        CgoCFLAGS    []string // cgo: flags for C compiler
        CgoCPPFLAGS  []string // cgo: flags for C preprocessor
        CgoCXXFLAGS  []string // cgo: flags for C++ compiler
        CgoFFLAGS    []string // cgo: flags for Fortran compiler
        CgoLDFLAGS   []string // cgo: flags for linker
        CgoPkgConfig []string // cgo: pkg-config names

        // Dependency information
        Imports      []string // import paths used by this package
        Deps         []string // all (recursively) imported dependencies
        TestImports  []string // imports from TestGoFiles
        XTestImports []string // imports from XTestGoFiles

        // Error information
        Incomplete bool            // this package or a dependency has an error
        Error      *PackageError   // error loading package
        DepsErrors []*PackageError // errors loading dependencies
    }

Packages stored in vendor directories report an ImportPath that includes the
path to the vendor directory (for example, "d/vendor/p" instead of "p"),
so that the ImportPath uniquely identifies a given copy of a package.
The Imports, Deps, TestImports, and XTestImports lists also contain these
expanded imports paths. See golang.org/s/go15vendor for more about vendoring.

The error information, if any, is

    type PackageError struct {
        ImportStack   []string // shortest path from package named on command line to this one
        Pos           string   // position of error (if present, file:line:col)
        Err           string   // the error itself
    }

The template function "join" calls strings.Join.

The template function "context" returns the build context, defined as:

	type Context struct {
		GOARCH        string   // target architecture
		GOOS          string   // target operating system
		GOROOT        string   // Go root
		GOPATH        string   // Go path
		CgoEnabled    bool     // whether cgo can be used
		UseAllFiles   bool     // use files regardless of +build lines, file names
		Compiler      string   // compiler to assume when computing target paths
		BuildTags     []string // build constraints to match in +build lines
		ReleaseTags   []string // releases the current release is compatible with
		InstallSuffix string   // suffix to use in the name of the install dir
	}

For more information about the meaning of these fields see the documentation
for the go/build package's Context type.

The -json flag causes the package data to be printed in JSON format
instead of using the template format.

The -cgo flag causes list to set CgoFiles not to the original *.go files
importing "C" but instead to the translated files generated by the cgo
command.

The -deps flag causes list to iterate over not just the named packages
but also all their dependencies. It visits them in a depth-first post-order
traversal, so that a package is listed only after all its dependencies.
Packages not explicitly listed on the command line will have the DepOnly
field set to true.

The -e flag changes the handling of erroneous packages, those that
cannot be found or are malformed. By default, the list command
prints an error to standard error for each erroneous package and
omits the packages from consideration during the usual printing.
With the -e flag, the list command never prints errors to standard
error and instead processes the erroneous packages with the usual
printing. Erroneous packages will have a non-empty ImportPath and
a non-nil Error field; other information may or may not be missing
(zeroed).

The -export flag causes list to set the Export field to the name of a
file containing up-to-date export information for the given package.

The -test flag causes list to report not only the named packages
but also their test binaries (for packages with tests), to convey to
source code analysis tools exactly how test binaries are constructed.
The reported import path for a test binary is the import path of
the package followed by a ".test" suffix, as in "math/rand.test".
When building a test, it is sometimes necessary to rebuild certain
dependencies specially for that test (most commonly the tested
package itself). The reported import path of a package recompiled
for a particular test binary is followed by a space and the name of
the test binary in brackets, as in "math/rand [math/rand.test]"
or "regexp [sort.test]". The ForTest field is also set to the name
of the package being tested ("math/rand" or "sort" in the previous
examples).

The Dir, Target, Shlib, Root, ConflictDir, and Export file paths
are all absolute paths.

By default, the lists GoFiles, CgoFiles, and so on hold names of files in Dir
(that is, paths relative to Dir, not absolute paths).
The extra entries added by the -cgo and -test flags are absolute paths
referring to cached copies of generated Go source files.
Although they are Go source files, the paths may not end in ".go".

For more about build flags, see 'go help build'.

For more about specifying packages, see 'go help packages'.
	`,
}

func init() {
	CmdList.Run = runList // break init cycle
	work.AddBuildFlags(CmdList)
}

var listCgo = CmdList.Flag.Bool("cgo", false, "")
var listDeps = CmdList.Flag.Bool("deps", false, "")
var listE = CmdList.Flag.Bool("e", false, "")
var listExport = CmdList.Flag.Bool("export", false, "")
var listFmt = CmdList.Flag.String("f", "{{.ImportPath}}", "")
var listJson = CmdList.Flag.Bool("json", false, "")
var listTest = CmdList.Flag.Bool("test", false, "")
var nl = []byte{'\n'}

func runList(cmd *base.Command, args []string) {
	work.BuildInit()
	out := newTrackingWriter(os.Stdout)
	defer out.w.Flush()

	var do func(*load.PackagePublic)
	if *listJson {
		do = func(p *load.PackagePublic) {
			b, err := json.MarshalIndent(p, "", "\t")
			if err != nil {
				out.Flush()
				base.Fatalf("%s", err)
			}
			out.Write(b)
			out.Write(nl)
		}
	} else {
		var cachedCtxt *Context
		context := func() *Context {
			if cachedCtxt == nil {
				cachedCtxt = newContext(&cfg.BuildContext)
			}
			return cachedCtxt
		}
		fm := template.FuncMap{
			"join":    strings.Join,
			"context": context,
		}
		tmpl, err := template.New("main").Funcs(fm).Parse(*listFmt)
		if err != nil {
			base.Fatalf("%s", err)
		}
		do = func(p *load.PackagePublic) {
			if err := tmpl.Execute(out, p); err != nil {
				out.Flush()
				base.Fatalf("%s", err)
			}
			if out.NeedNL() {
				out.Write(nl)
			}
		}
	}

	var pkgs []*load.Package
	if *listE {
		pkgs = load.PackagesAndErrors(args)
	} else {
		pkgs = load.Packages(args)
	}

	if cache.Default() == nil {
		// These flags return file names pointing into the build cache,
		// so the build cache must exist.
		if *listCgo {
			base.Fatalf("go list -cgo requires build cache")
		}
		if *listExport {
			base.Fatalf("go list -export requires build cache")
		}
		if *listTest {
			base.Fatalf("go list -test requires build cache")
		}
	}

	if *listTest {
		c := cache.Default()
		// Add test binaries to packages to be listed.
		for _, p := range pkgs {
			if p.Error != nil {
				continue
			}
			if len(p.TestGoFiles)+len(p.XTestGoFiles) > 0 {
				pmain, _, _, err := load.TestPackagesFor(p, nil)
				if err != nil {
					if !*listE {
						base.Errorf("can't load test package: %s", err)
						continue
					}
					pmain = &load.Package{
						PackagePublic: load.PackagePublic{
							ImportPath: p.ImportPath + ".test",
							Error:      &load.PackageError{Err: err.Error()},
						},
					}
				}
				pkgs = append(pkgs, pmain)

				data := *pmain.Internal.TestmainGo
				h := cache.NewHash("testmain")
				h.Write([]byte("testmain\n"))
				h.Write(data)
				out, _, err := c.Put(h.Sum(), bytes.NewReader(data))
				if err != nil {
					base.Fatalf("%s", err)
				}
				pmain.GoFiles[0] = c.OutputFile(out)
			}
		}
	}

	// Remember which packages are named on the command line.
	cmdline := make(map[*load.Package]bool)
	for _, p := range pkgs {
		cmdline[p] = true
	}

	if *listDeps {
		// Note: This changes the order of the listed packages
		// from "as written on the command line" to
		// "a depth-first post-order traversal".
		// (The dependency exploration order for a given node
		// is alphabetical, same as listed in .Deps.)
		// Note that -deps is applied after -test,
		// so that you only get descriptions of tests for the things named
		// explicitly on the command line, not for all dependencies.
		pkgs = load.PackageList(pkgs)
	}

	// Do we need to run a build to gather information?
	needStale := *listJson || strings.Contains(*listFmt, ".Stale")
	if needStale || *listExport || *listCgo {
		var b work.Builder
		b.Init()
		b.IsCmdList = true
		b.NeedExport = *listExport
		b.NeedCgoFiles = *listCgo
		a := &work.Action{}
		// TODO: Use pkgsFilter?
		for _, p := range pkgs {
			a.Deps = append(a.Deps, b.AutoAction(work.ModeInstall, work.ModeInstall, p))
		}
		b.Do(a)
	}

	for _, p := range pkgs {
		// Show vendor-expanded paths in listing
		p.TestImports = p.Resolve(p.TestImports)
		p.XTestImports = p.Resolve(p.XTestImports)
		p.DepOnly = !cmdline[p]
	}

	if *listTest {
		all := pkgs
		if !*listDeps {
			all = load.PackageList(pkgs)
		}
		// Update import paths to distinguish the real package p
		// from p recompiled for q.test.
		// This must happen only once the build code is done
		// looking at import paths, because it will get very confused
		// if it sees these.
		for _, p := range all {
			if p.ForTest != "" {
				p.ImportPath += " [" + p.ForTest + ".test]"
			}
			p.DepOnly = !cmdline[p]
		}
		// Update import path lists to use new strings.
		for _, p := range all {
			j := 0
			for i := range p.Imports {
				// Internal skips "C"
				if p.Imports[i] == "C" {
					continue
				}
				p.Imports[i] = p.Internal.Imports[j].ImportPath
				j++
			}
		}
		// Recompute deps lists using new strings, from the leaves up.
		for _, p := range all {
			deps := make(map[string]bool)
			for _, p1 := range p.Internal.Imports {
				deps[p1.ImportPath] = true
				for _, d := range p1.Deps {
					deps[d] = true
				}
			}
			p.Deps = make([]string, 0, len(deps))
			for d := range deps {
				p.Deps = append(p.Deps, d)
			}
			sort.Strings(p.Deps)
		}
	}

	for _, p := range pkgs {
		do(&p.PackagePublic)
	}
}

// TrackingWriter tracks the last byte written on every write so
// we can avoid printing a newline if one was already written or
// if there is no output at all.
type TrackingWriter struct {
	w    *bufio.Writer
	last byte
}

func newTrackingWriter(w io.Writer) *TrackingWriter {
	return &TrackingWriter{
		w:    bufio.NewWriter(w),
		last: '\n',
	}
}

func (t *TrackingWriter) Write(p []byte) (n int, err error) {
	n, err = t.w.Write(p)
	if n > 0 {
		t.last = p[n-1]
	}
	return
}

func (t *TrackingWriter) Flush() {
	t.w.Flush()
}

func (t *TrackingWriter) NeedNL() bool {
	return t.last != '\n'
}
