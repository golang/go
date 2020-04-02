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
	"cmd/go/internal/modload"
	"cmd/go/internal/str"
	"cmd/go/internal/work"
)

var CmdList = &base.Command{
	// Note: -f -json -m are listed explicitly because they are the most common list flags.
	// Do not send CLs removing them because they're covered by [list flags].
	UsageLine: "go list [-f format] [-json] [-m] [list flags] [build flags] [packages]",
	Short:     "list packages or modules",
	Long: `
List lists the named packages, one per line.
The most commonly-used flags are -f and -json, which control the form
of the output printed for each package. Other list flags, documented below,
control more specific details.

The default output shows the package import path:

    bytes
    encoding/json
    github.com/gorilla/mux
    golang.org/x/net/html

The -f flag specifies an alternate format for the list, using the
syntax of package template. The default output is equivalent
to -f '{{.ImportPath}}'. The struct being passed to the template is:

    type Package struct {
        Dir           string   // directory containing package sources
        ImportPath    string   // import path of package in dir
        ImportComment string   // path in import comment on package statement
        Name          string   // package name
        Doc           string   // package documentation string
        Target        string   // install path
        Shlib         string   // the shared library that contains this package (only set when -linkshared)
        Goroot        bool     // is this package in the Go root?
        Standard      bool     // is this package part of the standard Go library?
        Stale         bool     // would 'go install' do anything for this package?
        StaleReason   string   // explanation for Stale==true
        Root          string   // Go root or Go path dir containing this package
        ConflictDir   string   // this directory shadows Dir in $GOPATH
        BinaryOnly    bool     // binary-only package (no longer supported)
        ForTest       string   // package is only for use in named test
        Export        string   // file containing export data (when using -export)
        Module        *Module  // info about package's containing module, if any (can be nil)
        Match         []string // command-line patterns matching this package
        DepOnly       bool     // package is only a dependency, not explicitly listed

        // Source files
        GoFiles         []string // .go source files (excluding CgoFiles, TestGoFiles, XTestGoFiles)
        CgoFiles        []string // .go source files that import "C"
        CompiledGoFiles []string // .go files presented to compiler (when using -compiled)
        IgnoredGoFiles  []string // .go source files ignored due to build constraints
        CFiles          []string // .c source files
        CXXFiles        []string // .cc, .cxx and .cpp source files
        MFiles          []string // .m source files
        HFiles          []string // .h, .hh, .hpp and .hxx source files
        FFiles          []string // .f, .F, .for and .f90 Fortran source files
        SFiles          []string // .s source files
        SwigFiles       []string // .swig files
        SwigCXXFiles    []string // .swigcxx files
        SysoFiles       []string // .syso object files to add to archive
        TestGoFiles     []string // _test.go files in package
        XTestGoFiles    []string // _test.go files outside package

        // Cgo directives
        CgoCFLAGS    []string // cgo: flags for C compiler
        CgoCPPFLAGS  []string // cgo: flags for C preprocessor
        CgoCXXFLAGS  []string // cgo: flags for C++ compiler
        CgoFFLAGS    []string // cgo: flags for Fortran compiler
        CgoLDFLAGS   []string // cgo: flags for linker
        CgoPkgConfig []string // cgo: pkg-config names

        // Dependency information
        Imports      []string          // import paths used by this package
        ImportMap    map[string]string // map from source import to ImportPath (identity entries omitted)
        Deps         []string          // all (recursively) imported dependencies
        TestImports  []string          // imports from TestGoFiles
        XTestImports []string          // imports from XTestGoFiles

        // Error information
        Incomplete bool            // this package or a dependency has an error
        Error      *PackageError   // error loading package
        DepsErrors []*PackageError // errors loading dependencies
    }

Packages stored in vendor directories report an ImportPath that includes the
path to the vendor directory (for example, "d/vendor/p" instead of "p"),
so that the ImportPath uniquely identifies a given copy of a package.
The Imports, Deps, TestImports, and XTestImports lists also contain these
expanded import paths. See golang.org/s/go15vendor for more about vendoring.

The error information, if any, is

    type PackageError struct {
        ImportStack   []string // shortest path from package named on command line to this one
        Pos           string   // position of error (if present, file:line:col)
        Err           string   // the error itself
    }

The module information is a Module struct, defined in the discussion
of list -m below.

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

The -compiled flag causes list to set CompiledGoFiles to the Go source
files presented to the compiler. Typically this means that it repeats
the files listed in GoFiles and then also adds the Go code generated
by processing CgoFiles and SwigFiles. The Imports list contains the
union of all imports from both GoFiles and CompiledGoFiles.

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

The -find flag causes list to identify the named packages but not
resolve their dependencies: the Imports and Deps lists will be empty.

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
The generated files added when using the -compiled and -test flags
are absolute paths referring to cached copies of generated Go source files.
Although they are Go source files, the paths may not end in ".go".

The -m flag causes list to list modules instead of packages.

When listing modules, the -f flag still specifies a format template
applied to a Go struct, but now a Module struct:

    type Module struct {
        Path      string       // module path
        Version   string       // module version
        Versions  []string     // available module versions (with -versions)
        Replace   *Module      // replaced by this module
        Time      *time.Time   // time version was created
        Update    *Module      // available update, if any (with -u)
        Main      bool         // is this the main module?
        Indirect  bool         // is this module only an indirect dependency of main module?
        Dir       string       // directory holding files for this module, if any
        GoMod     string       // path to go.mod file used when loading this module, if any
        GoVersion string       // go version used in module
        Error     *ModuleError // error loading module
    }

    type ModuleError struct {
        Err string // the error itself
    }

The file GoMod refers to may be outside the module directory if the
module is in the module cache or if the -modfile flag is used.

The default output is to print the module path and then
information about the version and replacement if any.
For example, 'go list -m all' might print:

    my/main/module
    golang.org/x/text v0.3.0 => /tmp/text
    rsc.io/pdf v0.1.1

The Module struct has a String method that formats this
line of output, so that the default format is equivalent
to -f '{{.String}}'.

Note that when a module has been replaced, its Replace field
describes the replacement module, and its Dir field is set to
the replacement's source code, if present. (That is, if Replace
is non-nil, then Dir is set to Replace.Dir, with no access to
the replaced source code.)

The -u flag adds information about available upgrades.
When the latest version of a given module is newer than
the current one, list -u sets the Module's Update field
to information about the newer module.
The Module's String method indicates an available upgrade by
formatting the newer version in brackets after the current version.
For example, 'go list -m -u all' might print:

    my/main/module
    golang.org/x/text v0.3.0 [v0.4.0] => /tmp/text
    rsc.io/pdf v0.1.1 [v0.1.2]

(For tools, 'go list -m -u -json all' may be more convenient to parse.)

The -versions flag causes list to set the Module's Versions field
to a list of all known versions of that module, ordered according
to semantic versioning, earliest to latest. The flag also changes
the default output format to display the module path followed by the
space-separated version list.

The arguments to list -m are interpreted as a list of modules, not packages.
The main module is the module containing the current directory.
The active modules are the main module and its dependencies.
With no arguments, list -m shows the main module.
With arguments, list -m shows the modules specified by the arguments.
Any of the active modules can be specified by its module path.
The special pattern "all" specifies all the active modules, first the main
module and then dependencies sorted by module path.
A pattern containing "..." specifies the active modules whose
module paths match the pattern.
A query of the form path@version specifies the result of that query,
which is not limited to active modules.
See 'go help modules' for more about module queries.

The template function "module" takes a single string argument
that must be a module path or query and returns the specified
module as a Module struct. If an error occurs, the result will
be a Module struct with a non-nil Error field.

For more about build flags, see 'go help build'.

For more about specifying packages, see 'go help packages'.

For more about modules, see 'go help modules'.
	`,
}

func init() {
	CmdList.Run = runList // break init cycle
	work.AddBuildFlags(CmdList, work.DefaultBuildFlags)
}

var (
	listCompiled = CmdList.Flag.Bool("compiled", false, "")
	listDeps     = CmdList.Flag.Bool("deps", false, "")
	listE        = CmdList.Flag.Bool("e", false, "")
	listExport   = CmdList.Flag.Bool("export", false, "")
	listFmt      = CmdList.Flag.String("f", "", "")
	listFind     = CmdList.Flag.Bool("find", false, "")
	listJson     = CmdList.Flag.Bool("json", false, "")
	listM        = CmdList.Flag.Bool("m", false, "")
	listU        = CmdList.Flag.Bool("u", false, "")
	listTest     = CmdList.Flag.Bool("test", false, "")
	listVersions = CmdList.Flag.Bool("versions", false, "")
)

var nl = []byte{'\n'}

func runList(cmd *base.Command, args []string) {
	modload.LoadTests = *listTest
	work.BuildInit()
	out := newTrackingWriter(os.Stdout)
	defer out.w.Flush()

	if *listFmt == "" {
		if *listM {
			*listFmt = "{{.String}}"
			if *listVersions {
				*listFmt = `{{.Path}}{{range .Versions}} {{.}}{{end}}`
			}
		} else {
			*listFmt = "{{.ImportPath}}"
		}
	}

	var do func(interface{})
	if *listJson {
		do = func(x interface{}) {
			b, err := json.MarshalIndent(x, "", "\t")
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
			"module":  modload.ModuleInfo,
		}
		tmpl, err := template.New("main").Funcs(fm).Parse(*listFmt)
		if err != nil {
			base.Fatalf("%s", err)
		}
		do = func(x interface{}) {
			if err := tmpl.Execute(out, x); err != nil {
				out.Flush()
				base.Fatalf("%s", err)
			}
			if out.NeedNL() {
				out.Write(nl)
			}
		}
	}

	if *listM {
		// Module mode.
		if *listCompiled {
			base.Fatalf("go list -compiled cannot be used with -m")
		}
		if *listDeps {
			// TODO(rsc): Could make this mean something with -m.
			base.Fatalf("go list -deps cannot be used with -m")
		}
		if *listExport {
			base.Fatalf("go list -export cannot be used with -m")
		}
		if *listFind {
			base.Fatalf("go list -find cannot be used with -m")
		}
		if *listTest {
			base.Fatalf("go list -test cannot be used with -m")
		}

		if modload.Init(); !modload.Enabled() {
			base.Fatalf("go list -m: not using modules")
		}

		modload.InitMod() // Parses go.mod and sets cfg.BuildMod.
		if cfg.BuildMod == "vendor" {
			const actionDisabledFormat = "go list -m: can't %s using the vendor directory\n\t(Use -mod=mod or -mod=readonly to bypass.)"

			if *listVersions {
				base.Fatalf(actionDisabledFormat, "determine available versions")
			}
			if *listU {
				base.Fatalf(actionDisabledFormat, "determine available upgrades")
			}

			for _, arg := range args {
				// In vendor mode, the module graph is incomplete: it contains only the
				// explicit module dependencies and the modules that supply packages in
				// the import graph. Reject queries that imply more information than that.
				if arg == "all" {
					base.Fatalf(actionDisabledFormat, "compute 'all'")
				}
				if strings.Contains(arg, "...") {
					base.Fatalf(actionDisabledFormat, "match module patterns")
				}
			}
		}

		modload.LoadBuildList()

		mods := modload.ListModules(args, *listU, *listVersions)
		if !*listE {
			for _, m := range mods {
				if m.Error != nil {
					base.Errorf("go list -m: %v", m.Error.Err)
				}
			}
			base.ExitIfErrors()
		}
		for _, m := range mods {
			do(m)
		}
		return
	}

	// Package mode (not -m).
	if *listU {
		base.Fatalf("go list -u can only be used with -m")
	}
	if *listVersions {
		base.Fatalf("go list -versions can only be used with -m")
	}

	// These pairings make no sense.
	if *listFind && *listDeps {
		base.Fatalf("go list -deps cannot be used with -find")
	}
	if *listFind && *listTest {
		base.Fatalf("go list -test cannot be used with -find")
	}

	load.IgnoreImports = *listFind
	var pkgs []*load.Package
	if *listE {
		pkgs = load.PackagesAndErrors(args)
	} else {
		pkgs = load.Packages(args)
		base.ExitIfErrors()
	}

	if cache.Default() == nil {
		// These flags return file names pointing into the build cache,
		// so the build cache must exist.
		if *listCompiled {
			base.Fatalf("go list -compiled requires build cache")
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
			if len(p.TestGoFiles)+len(p.XTestGoFiles) > 0 {
				var pmain, ptest, pxtest *load.Package
				var err error
				if *listE {
					pmain, ptest, pxtest = load.TestPackagesAndErrors(p, nil)
				} else {
					pmain, ptest, pxtest, err = load.TestPackagesFor(p, nil)
					if err != nil {
						base.Errorf("can't load test package: %s", err)
					}
				}
				if pmain != nil {
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
				if ptest != nil && ptest != p {
					pkgs = append(pkgs, ptest)
				}
				if pxtest != nil {
					pkgs = append(pkgs, pxtest)
				}
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
	if needStale || *listExport || *listCompiled {
		var b work.Builder
		b.Init()
		b.IsCmdList = true
		b.NeedExport = *listExport
		b.NeedCompiledGoFiles = *listCompiled
		a := &work.Action{}
		// TODO: Use pkgsFilter?
		for _, p := range pkgs {
			if len(p.GoFiles)+len(p.CgoFiles) > 0 {
				a.Deps = append(a.Deps, b.AutoAction(work.ModeInstall, work.ModeInstall, p))
			}
		}
		b.Do(a)
	}

	for _, p := range pkgs {
		// Show vendor-expanded paths in listing
		p.TestImports = p.Resolve(p.TestImports)
		p.XTestImports = p.Resolve(p.XTestImports)
		p.DepOnly = !cmdline[p]

		if *listCompiled {
			p.Imports = str.StringList(p.Imports, p.Internal.CompiledImports)
		}
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
		old := make(map[string]string)
		for _, p := range all {
			if p.ForTest != "" {
				new := p.ImportPath + " [" + p.ForTest + ".test]"
				old[new] = p.ImportPath
				p.ImportPath = new
			}
			p.DepOnly = !cmdline[p]
		}
		// Update import path lists to use new strings.
		m := make(map[string]string)
		for _, p := range all {
			for _, p1 := range p.Internal.Imports {
				if p1.ForTest != "" {
					m[old[p1.ImportPath]] = p1.ImportPath
				}
			}
			for i, old := range p.Imports {
				if new := m[old]; new != "" {
					p.Imports[i] = new
				}
			}
			for old := range m {
				delete(m, old)
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

	// Record non-identity import mappings in p.ImportMap.
	for _, p := range pkgs {
		for i, srcPath := range p.Internal.RawImports {
			path := p.Imports[i]
			if path != srcPath {
				if p.ImportMap == nil {
					p.ImportMap = make(map[string]string)
				}
				p.ImportMap[srcPath] = path
			}
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
