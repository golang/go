// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package list implements the “go list” command.
package list

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"reflect"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"text/template"

	"golang.org/x/sync/semaphore"

	"cmd/go/internal/base"
	"cmd/go/internal/cache"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/modinfo"
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
        Dir            string   // directory containing package sources
        ImportPath     string   // import path of package in dir
        ImportComment  string   // path in import comment on package statement
        Name           string   // package name
        Doc            string   // package documentation string
        Target         string   // install path
        Shlib          string   // the shared library that contains this package (only set when -linkshared)
        Goroot         bool     // is this package in the Go root?
        Standard       bool     // is this package part of the standard Go library?
        Stale          bool     // would 'go install' do anything for this package?
        StaleReason    string   // explanation for Stale==true
        Root           string   // Go root or Go path dir containing this package
        ConflictDir    string   // this directory shadows Dir in $GOPATH
        BinaryOnly     bool     // binary-only package (no longer supported)
        ForTest        string   // package is only for use in named test
        Export         string   // file containing export data (when using -export)
        BuildID        string   // build ID of the compiled package (when using -export)
        Module         *Module  // info about package's containing module, if any (can be nil)
        Match          []string // command-line patterns matching this package
        DepOnly        bool     // package is only a dependency, not explicitly listed
        DefaultGODEBUG string  // default GODEBUG setting, for main packages

        // Source files
        GoFiles           []string   // .go source files (excluding CgoFiles, TestGoFiles, XTestGoFiles)
        CgoFiles          []string   // .go source files that import "C"
        CompiledGoFiles   []string   // .go files presented to compiler (when using -compiled)
        IgnoredGoFiles    []string   // .go source files ignored due to build constraints
        IgnoredOtherFiles []string // non-.go source files ignored due to build constraints
        CFiles            []string   // .c source files
        CXXFiles          []string   // .cc, .cxx and .cpp source files
        MFiles            []string   // .m source files
        HFiles            []string   // .h, .hh, .hpp and .hxx source files
        FFiles            []string   // .f, .F, .for and .f90 Fortran source files
        SFiles            []string   // .s source files
        SwigFiles         []string   // .swig files
        SwigCXXFiles      []string   // .swigcxx files
        SysoFiles         []string   // .syso object files to add to archive
        TestGoFiles       []string   // _test.go files in package
        XTestGoFiles      []string   // _test.go files outside package

        // Embedded files
        EmbedPatterns      []string // //go:embed patterns
        EmbedFiles         []string // files matched by EmbedPatterns
        TestEmbedPatterns  []string // //go:embed patterns in TestGoFiles
        TestEmbedFiles     []string // files matched by TestEmbedPatterns
        XTestEmbedPatterns []string // //go:embed patterns in XTestGoFiles
        XTestEmbedFiles    []string // files matched by XTestEmbedPatterns

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
        UseAllFiles   bool     // use files regardless of //go:build lines, file names
        Compiler      string   // compiler to assume when computing target paths
        BuildTags     []string // build constraints to match in //go:build lines
        ToolTags      []string // toolchain-specific build constraints
        ReleaseTags   []string // releases the current release is compatible with
        InstallSuffix string   // suffix to use in the name of the install dir
    }

For more information about the meaning of these fields see the documentation
for the go/build package's Context type.

The -json flag causes the package data to be printed in JSON format
instead of using the template format. The JSON flag can optionally be
provided with a set of comma-separated required field names to be output.
If so, those required fields will always appear in JSON output, but
others may be omitted to save work in computing the JSON struct.

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
file containing up-to-date export information for the given package,
and the BuildID field to the build ID of the compiled package.

The -find flag causes list to identify the named packages but not
resolve their dependencies: the Imports and Deps lists will be empty.
With the -find flag, the -deps, -test and -export commands cannot be
used.

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
        Path       string        // module path
        Query      string        // version query corresponding to this version
        Version    string        // module version
        Versions   []string      // available module versions
        Replace    *Module       // replaced by this module
        Time       *time.Time    // time version was created
        Update     *Module       // available update (with -u)
        Main       bool          // is this the main module?
        Indirect   bool          // module is only indirectly needed by main module
        Dir        string        // directory holding local copy of files, if any
        GoMod      string        // path to go.mod file describing module, if any
        GoVersion  string        // go version used in module
        Retracted  []string      // retraction information, if any (with -retracted or -u)
        Deprecated string        // deprecation message, if any (with -u)
        Error      *ModuleError  // error loading module
        Sum        string        // checksum for path, version (as in go.sum)
        GoModSum   string        // checksum for go.mod (as in go.sum)
        Origin     any           // provenance of module
        Reuse      bool          // reuse of old module info is safe
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
to information about the newer module. list -u will also set
the module's Retracted field if the current version is retracted.
The Module's String method indicates an available upgrade by
formatting the newer version in brackets after the current version.
If a version is retracted, the string "(retracted)" will follow it.
For example, 'go list -m -u all' might print:

    my/main/module
    golang.org/x/text v0.3.0 [v0.4.0] => /tmp/text
    rsc.io/pdf v0.1.1 (retracted) [v0.1.2]

(For tools, 'go list -m -u -json all' may be more convenient to parse.)

The -versions flag causes list to set the Module's Versions field
to a list of all known versions of that module, ordered according
to semantic versioning, earliest to latest. The flag also changes
the default output format to display the module path followed by the
space-separated version list.

The -retracted flag causes list to report information about retracted
module versions. When -retracted is used with -f or -json, the Retracted
field explains why the version was retracted.
The strings are taken from comments on the retract directive in the
module's go.mod file. When -retracted is used with -versions, retracted
versions are listed together with unretracted versions. The -retracted
flag may be used with or without -m.

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

When using -m, the -reuse=old.json flag accepts the name of file containing
the JSON output of a previous 'go list -m -json' invocation with the
same set of modifier flags (such as -u, -retracted, and -versions).
The go command may use this file to determine that a module is unchanged
since the previous invocation and avoid redownloading information about it.
Modules that are not redownloaded will be marked in the new output by
setting the Reuse field to true. Normally the module cache provides this
kind of reuse automatically; the -reuse flag can be useful on systems that
do not preserve the module cache.

For more about build flags, see 'go help build'.

For more about specifying packages, see 'go help packages'.

For more about modules, see https://golang.org/ref/mod.
	`,
}

func init() {
	CmdList.Run = runList // break init cycle
	// Omit build -json because list has its own -json
	work.AddBuildFlags(CmdList, work.OmitJSONFlag)
	work.AddCoverFlags(CmdList, nil)
	CmdList.Flag.Var(&listJsonFields, "json", "")
}

var (
	listCompiled   = CmdList.Flag.Bool("compiled", false, "")
	listDeps       = CmdList.Flag.Bool("deps", false, "")
	listE          = CmdList.Flag.Bool("e", false, "")
	listExport     = CmdList.Flag.Bool("export", false, "")
	listFmt        = CmdList.Flag.String("f", "", "")
	listFind       = CmdList.Flag.Bool("find", false, "")
	listJson       bool
	listJsonFields jsonFlag // If not empty, only output these fields.
	listM          = CmdList.Flag.Bool("m", false, "")
	listRetracted  = CmdList.Flag.Bool("retracted", false, "")
	listReuse      = CmdList.Flag.String("reuse", "", "")
	listTest       = CmdList.Flag.Bool("test", false, "")
	listU          = CmdList.Flag.Bool("u", false, "")
	listVersions   = CmdList.Flag.Bool("versions", false, "")
)

// A StringsFlag is a command-line flag that interprets its argument
// as a space-separated list of possibly-quoted strings.
type jsonFlag map[string]bool

func (v *jsonFlag) Set(s string) error {
	if v, err := strconv.ParseBool(s); err == nil {
		listJson = v
		return nil
	}
	listJson = true
	if *v == nil {
		*v = make(map[string]bool)
	}
	for f := range strings.SplitSeq(s, ",") {
		(*v)[f] = true
	}
	return nil
}

func (v *jsonFlag) String() string {
	fields := make([]string, 0, len(*v))
	for f := range *v {
		fields = append(fields, f)
	}
	sort.Strings(fields)
	return strings.Join(fields, ",")
}

func (v *jsonFlag) IsBoolFlag() bool {
	return true
}

func (v *jsonFlag) needAll() bool {
	return len(*v) == 0
}

func (v *jsonFlag) needAny(fields ...string) bool {
	if v.needAll() {
		return true
	}
	for _, f := range fields {
		if (*v)[f] {
			return true
		}
	}
	return false
}

var nl = []byte{'\n'}

func runList(ctx context.Context, cmd *base.Command, args []string) {
	modload.InitWorkfile(modload.LoaderState)

	if *listFmt != "" && listJson {
		base.Fatalf("go list -f cannot be used with -json")
	}
	if *listReuse != "" && !*listM {
		base.Fatalf("go list -reuse cannot be used without -m")
	}
	if *listReuse != "" && modload.HasModRoot(modload.LoaderState) {
		base.Fatalf("go list -reuse cannot be used inside a module")
	}

	work.BuildInit(modload.LoaderState)
	out := newTrackingWriter(os.Stdout)
	defer out.w.Flush()

	if *listFmt == "" {
		if *listM {
			*listFmt = "{{.String}}"
			if *listVersions {
				*listFmt = `{{.Path}}{{range .Versions}} {{.}}{{end}}{{if .Deprecated}} (deprecated){{end}}`
			}
		} else {
			*listFmt = "{{.ImportPath}}"
		}
	}

	var do func(x any)
	if listJson {
		do = func(x any) {
			if !listJsonFields.needAll() {
				//  Set x to a copy of itself with all non-requested fields cleared.
				v := reflect.New(reflect.TypeOf(x).Elem()).Elem() // do is always called with a non-nil pointer.
				v.Set(reflect.ValueOf(x).Elem())
				for i := 0; i < v.NumField(); i++ {
					if !listJsonFields.needAny(v.Type().Field(i).Name) {
						v.Field(i).SetZero()
					}
				}
				x = v.Interface()
			}
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
			"module":  func(path string) *modinfo.ModulePublic { return modload.ModuleInfo(modload.LoaderState, ctx, path) },
		}
		tmpl, err := template.New("main").Funcs(fm).Parse(*listFmt)
		if err != nil {
			base.Fatalf("%s", err)
		}
		do = func(x any) {
			if err := tmpl.Execute(out, x); err != nil {
				out.Flush()
				base.Fatalf("%s", err)
			}
			if out.NeedNL() {
				out.Write(nl)
			}
		}
	}

	modload.Init(modload.LoaderState)
	if *listRetracted {
		if cfg.BuildMod == "vendor" {
			base.Fatalf("go list -retracted cannot be used when vendoring is enabled")
		}
		if !modload.Enabled(modload.LoaderState) {
			base.Fatalf("go list -retracted can only be used in module-aware mode")
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

		if modload.Init(modload.LoaderState); !modload.Enabled(modload.LoaderState) {
			base.Fatalf("go: list -m cannot be used with GO111MODULE=off")
		}

		modload.LoadModFile(modload.LoaderState, ctx) // Sets cfg.BuildMod as a side-effect.
		if cfg.BuildMod == "vendor" {
			const actionDisabledFormat = "go: can't %s using the vendor directory\n\t(Use -mod=mod or -mod=readonly to bypass.)"

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

		var mode modload.ListMode
		if *listU {
			mode |= modload.ListU | modload.ListRetracted | modload.ListDeprecated
		}
		if *listRetracted {
			mode |= modload.ListRetracted
		}
		if *listVersions {
			mode |= modload.ListVersions
			if *listRetracted {
				mode |= modload.ListRetractedVersions
			}
		}
		if *listReuse != "" && len(args) == 0 {
			base.Fatalf("go: list -m -reuse only has an effect with module@version arguments")
		}
		mods, err := modload.ListModules(modload.LoaderState, ctx, args, mode, *listReuse)
		if !*listE {
			for _, m := range mods {
				if m.Error != nil {
					base.Error(errors.New(m.Error.Err))
				}
			}
			if err != nil {
				base.Error(err)
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
	if *listFind && *listExport {
		base.Fatalf("go list -export cannot be used with -find")
	}

	pkgOpts := load.PackageOpts{
		IgnoreImports:      *listFind,
		ModResolveTests:    *listTest,
		AutoVCS:            true,
		SuppressBuildInfo:  !*listExport && !listJsonFields.needAny("Stale", "StaleReason"),
		SuppressEmbedFiles: !*listExport && !listJsonFields.needAny("EmbedFiles", "TestEmbedFiles", "XTestEmbedFiles"),
	}
	pkgs := load.PackagesAndErrors(modload.LoaderState, ctx, pkgOpts, args)
	if !*listE {
		w := 0
		for _, pkg := range pkgs {
			if pkg.Error != nil {
				base.Errorf("%v", pkg.Error)
				continue
			}
			pkgs[w] = pkg
			w++
		}
		pkgs = pkgs[:w]
		base.ExitIfErrors()
	}

	if *listTest {
		c := cache.Default()
		// Add test binaries to packages to be listed.

		var wg sync.WaitGroup
		sema := semaphore.NewWeighted(int64(runtime.GOMAXPROCS(0)))
		type testPackageSet struct {
			p, pmain, ptest, pxtest *load.Package
		}
		var testPackages []testPackageSet
		for _, p := range pkgs {
			if len(p.TestGoFiles)+len(p.XTestGoFiles) > 0 {
				var pmain, ptest, pxtest *load.Package
				if *listE {
					sema.Acquire(ctx, 1)
					wg.Add(1)
					done := func() {
						sema.Release(1)
						wg.Done()
					}
					pmain, ptest, pxtest = load.TestPackagesAndErrors(modload.LoaderState, ctx, done, pkgOpts, p, nil)
				} else {
					var perr *load.Package
					pmain, ptest, pxtest, perr = load.TestPackagesFor(modload.LoaderState, ctx, pkgOpts, p, nil)
					if perr != nil {
						base.Fatalf("go: can't load test package: %s", perr.Error)
					}
				}
				testPackages = append(testPackages, testPackageSet{p, pmain, ptest, pxtest})
			}
		}
		wg.Wait()
		for _, pkgset := range testPackages {
			p, pmain, ptest, pxtest := pkgset.p, pkgset.pmain, pkgset.ptest, pkgset.pxtest
			if pmain != nil {
				pkgs = append(pkgs, pmain)
				data := *pmain.Internal.TestmainGo
				sema.Acquire(ctx, 1)
				wg.Add(1)
				go func() {
					h := cache.NewHash("testmain")
					h.Write([]byte("testmain\n"))
					h.Write(data)
					out, _, err := c.Put(h.Sum(), bytes.NewReader(data))
					if err != nil {
						base.Fatalf("%s", err)
					}
					pmain.GoFiles[0] = c.OutputFile(out)
					sema.Release(1)
					wg.Done()
				}()

			}
			if ptest != nil && ptest != p {
				pkgs = append(pkgs, ptest)
			}
			if pxtest != nil {
				pkgs = append(pkgs, pxtest)
			}
		}

		wg.Wait()
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
		pkgs = loadPackageList(pkgs)
	}

	// Do we need to run a build to gather information?
	needStale := (listJson && listJsonFields.needAny("Stale", "StaleReason")) || strings.Contains(*listFmt, ".Stale")
	if needStale || *listExport || *listCompiled {
		b := work.NewBuilder("", modload.LoaderState.VendorDirOrEmpty)
		if *listE {
			b.AllowErrors = true
		}
		defer func() {
			if err := b.Close(); err != nil {
				base.Fatal(err)
			}
		}()

		b.IsCmdList = true
		b.NeedExport = *listExport
		b.NeedCompiledGoFiles = *listCompiled
		if cfg.BuildCover {
			load.PrepareForCoverageBuild(modload.LoaderState, pkgs)
		}
		a := &work.Action{}
		// TODO: Use pkgsFilter?
		for _, p := range pkgs {
			if len(p.GoFiles)+len(p.CgoFiles) > 0 {
				a.Deps = append(a.Deps, b.AutoAction(modload.LoaderState, work.ModeInstall, work.ModeInstall, p))
			}
		}
		b.Do(ctx, a)
	}

	for _, p := range pkgs {
		// Show vendor-expanded paths in listing
		p.TestImports = p.Resolve(modload.LoaderState, p.TestImports)
		p.XTestImports = p.Resolve(modload.LoaderState, p.XTestImports)
		p.DepOnly = !cmdline[p]

		if *listCompiled {
			p.Imports = str.StringList(p.Imports, p.Internal.CompiledImports)
		}
	}

	if *listTest || (cfg.BuildPGO == "auto" && len(cmdline) > 1) {
		all := pkgs
		if !*listDeps {
			all = loadPackageList(pkgs)
		}
		// Update import paths to distinguish the real package p
		// from p recompiled for q.test, or to distinguish between
		// p compiled with different PGO profiles.
		// This must happen only once the build code is done
		// looking at import paths, because it will get very confused
		// if it sees these.
		old := make(map[string]string)
		for _, p := range all {
			if p.ForTest != "" || p.Internal.ForMain != "" {
				new := p.Desc()
				old[new] = p.ImportPath
				p.ImportPath = new
			}
			p.DepOnly = !cmdline[p]
		}
		// Update import path lists to use new strings.
		m := make(map[string]string)
		for _, p := range all {
			for _, p1 := range p.Internal.Imports {
				if p1.ForTest != "" || p1.Internal.ForMain != "" {
					m[old[p1.ImportPath]] = p1.ImportPath
				}
			}
			for i, old := range p.Imports {
				if new := m[old]; new != "" {
					p.Imports[i] = new
				}
			}
			clear(m)
		}
	}

	if listJsonFields.needAny("Deps", "DepsErrors") {
		all := pkgs
		// Make sure we iterate through packages in a postorder traversal,
		// which load.PackageList guarantees. If *listDeps, then all is
		// already in PackageList order. Otherwise, calling load.PackageList
		// provides the guarantee. In the case of an import cycle, the last package
		// visited in the cycle, importing the first encountered package in the cycle,
		// is visited first. The cycle import error will be bubbled up in the traversal
		// order up to the first package in the cycle, covering all the packages
		// in the cycle.
		if !*listDeps {
			all = load.PackageList(pkgs)
		}
		if listJsonFields.needAny("Deps") {
			for _, p := range all {
				collectDeps(p)
			}
		}
		if listJsonFields.needAny("DepsErrors") {
			for _, p := range all {
				collectDepsErrors(p)
			}
		}
	}

	// TODO(golang.org/issue/40676): This mechanism could be extended to support
	// -u without -m.
	if *listRetracted {
		// Load retractions for modules that provide packages that will be printed.
		// TODO(golang.org/issue/40775): Packages from the same module refer to
		// distinct ModulePublic instance. It would be nice if they could all point
		// to the same instance. This would require additional global state in
		// modload.loaded, so that should be refactored first. For now, we update
		// all instances.
		modToArg := make(map[*modinfo.ModulePublic]string)
		argToMods := make(map[string][]*modinfo.ModulePublic)
		var args []string
		addModule := func(mod *modinfo.ModulePublic) {
			if mod.Version == "" {
				return
			}
			arg := fmt.Sprintf("%s@%s", mod.Path, mod.Version)
			if argToMods[arg] == nil {
				args = append(args, arg)
			}
			argToMods[arg] = append(argToMods[arg], mod)
			modToArg[mod] = arg
		}
		for _, p := range pkgs {
			if p.Module == nil {
				continue
			}
			addModule(p.Module)
			if p.Module.Replace != nil {
				addModule(p.Module.Replace)
			}
		}

		if len(args) > 0 {
			var mode modload.ListMode
			if *listRetracted {
				mode |= modload.ListRetracted
			}
			rmods, err := modload.ListModules(modload.LoaderState, ctx, args, mode, *listReuse)
			if err != nil && !*listE {
				base.Error(err)
			}
			for i, arg := range args {
				rmod := rmods[i]
				for _, mod := range argToMods[arg] {
					mod.Retracted = rmod.Retracted
					if rmod.Error != nil && mod.Error == nil {
						mod.Error = rmod.Error
					}
				}
			}
		}
	}

	// Record non-identity import mappings in p.ImportMap.
	for _, p := range pkgs {
		nRaw := len(p.Internal.RawImports)
		for i, path := range p.Imports {
			var srcPath string
			if i < nRaw {
				srcPath = p.Internal.RawImports[i]
			} else {
				// This path is not within the raw imports, so it must be an import
				// found only within CompiledGoFiles. Those paths are found in
				// CompiledImports.
				srcPath = p.Internal.CompiledImports[i-nRaw]
			}

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

// loadPackageList is like load.PackageList, but prints error messages and exits
// with nonzero status if listE is not set and any package in the expanded list
// has errors.
func loadPackageList(roots []*load.Package) []*load.Package {
	pkgs := load.PackageList(roots)

	if !*listE {
		for _, pkg := range pkgs {
			if pkg.Error != nil {
				base.Errorf("%v", pkg.Error)
			}
		}
	}

	return pkgs
}

// collectDeps populates p.Deps by iterating over p.Internal.Imports.
// collectDeps must be called on all of p's Imports before being called on p.
func collectDeps(p *load.Package) {
	deps := make(map[string]bool)

	for _, p := range p.Internal.Imports {
		deps[p.ImportPath] = true
		for _, q := range p.Deps {
			deps[q] = true
		}
	}

	p.Deps = make([]string, 0, len(deps))
	for dep := range deps {
		p.Deps = append(p.Deps, dep)
	}
	sort.Strings(p.Deps)
}

// collectDepsErrors populates p.DepsErrors by iterating over p.Internal.Imports.
// collectDepsErrors must be called on all of p's Imports before being called on p.
func collectDepsErrors(p *load.Package) {
	depsErrors := make(map[*load.PackageError]bool)

	for _, p := range p.Internal.Imports {
		if p.Error != nil {
			depsErrors[p.Error] = true
		}
		for _, q := range p.DepsErrors {
			depsErrors[q] = true
		}
	}

	p.DepsErrors = make([]*load.PackageError, 0, len(depsErrors))
	for deperr := range depsErrors {
		p.DepsErrors = append(p.DepsErrors, deperr)
	}
	// Sort packages by the package on the top of the stack, which should be
	// the package the error was produced for. Each package can have at most
	// one error set on it.
	sort.Slice(p.DepsErrors, func(i, j int) bool {
		stki, stkj := p.DepsErrors[i].ImportStack, p.DepsErrors[j].ImportStack
		// Some packages are missing import stacks. To ensure deterministic
		// sort order compare two errors that are missing import stacks by
		// their errors' error texts.
		if len(stki) == 0 {
			if len(stkj) != 0 {
				return true
			}

			return p.DepsErrors[i].Err.Error() < p.DepsErrors[j].Err.Error()
		} else if len(stkj) == 0 {
			return false
		}
		pathi, pathj := stki[len(stki)-1], stkj[len(stkj)-1]
		return pathi.Pkg < pathj.Pkg
	})
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
