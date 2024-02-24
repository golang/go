// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"go/build"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/search"
	"cmd/go/internal/trace"
)

var CmdBuild = &base.Command{
	UsageLine: "go build [-o output] [build flags] [packages]",
	Short:     "compile packages and dependencies",
	Long: `
Build compiles the packages named by the import paths,
along with their dependencies, but it does not install the results.

If the arguments to build are a list of .go files from a single directory,
build treats them as a list of source files specifying a single package.

When compiling packages, build ignores files that end in '_test.go'.

When compiling a single main package, build writes the resulting
executable to an output file named after the last non-major-version
component of the package import path. The '.exe' suffix is added
when writing a Windows executable.
So 'go build example/sam' writes 'sam' or 'sam.exe'.
'go build example.com/foo/v2' writes 'foo' or 'foo.exe', not 'v2.exe'.

When compiling a package from a list of .go files, the executable
is named after the first source file.
'go build ed.go rx.go' writes 'ed' or 'ed.exe'.

When compiling multiple packages or a single non-main package,
build compiles the packages but discards the resulting object,
serving only as a check that the packages can be built.

The -o flag forces build to write the resulting executable or object
to the named output file or directory, instead of the default behavior described
in the last two paragraphs. If the named output is an existing directory or
ends with a slash or backslash, then any resulting executables
will be written to that directory.

The build flags are shared by the build, clean, get, install, list, run,
and test commands:

	-C dir
		Change to dir before running the command.
		Any files named on the command line are interpreted after
		changing directories.
		If used, this flag must be the first one in the command line.
	-a
		force rebuilding of packages that are already up-to-date.
	-n
		print the commands but do not run them.
	-p n
		the number of programs, such as build commands or
		test binaries, that can be run in parallel.
		The default is GOMAXPROCS, normally the number of CPUs available.
	-race
		enable data race detection.
		Supported only on linux/amd64, freebsd/amd64, darwin/amd64, darwin/arm64, windows/amd64,
		linux/ppc64le and linux/arm64 (only for 48-bit VMA).
	-msan
		enable interoperation with memory sanitizer.
		Supported only on linux/amd64, linux/arm64, linux/loong64, freebsd/amd64
		and only with Clang/LLVM as the host C compiler.
		PIE build mode will be used on all platforms except linux/amd64.
	-asan
		enable interoperation with address sanitizer.
		Supported only on linux/arm64, linux/amd64, linux/loong64.
		Supported on linux/amd64 or linux/arm64 and only with GCC 7 and higher
		or Clang/LLVM 9 and higher.
		And supported on linux/loong64 only with Clang/LLVM 16 and higher.
	-cover
		enable code coverage instrumentation.
	-covermode set,count,atomic
		set the mode for coverage analysis.
		The default is "set" unless -race is enabled,
		in which case it is "atomic".
		The values:
		set: bool: does this statement run?
		count: int: how many times does this statement run?
		atomic: int: count, but correct in multithreaded tests;
			significantly more expensive.
		Sets -cover.
	-coverpkg pattern1,pattern2,pattern3
		For a build that targets package 'main' (e.g. building a Go
		executable), apply coverage analysis to each package matching
		the patterns. The default is to apply coverage analysis to
		packages in the main Go module. See 'go help packages' for a
		description of package patterns.  Sets -cover.
	-v
		print the names of packages as they are compiled.
	-work
		print the name of the temporary work directory and
		do not delete it when exiting.
	-x
		print the commands.
	-asmflags '[pattern=]arg list'
		arguments to pass on each go tool asm invocation.
	-buildmode mode
		build mode to use. See 'go help buildmode' for more.
	-buildvcs
		Whether to stamp binaries with version control information
		("true", "false", or "auto"). By default ("auto"), version control
		information is stamped into a binary if the main package, the main module
		containing it, and the current directory are all in the same repository.
		Use -buildvcs=false to always omit version control information, or
		-buildvcs=true to error out if version control information is available but
		cannot be included due to a missing tool or ambiguous directory structure.
	-compiler name
		name of compiler to use, as in runtime.Compiler (gccgo or gc).
	-gccgoflags '[pattern=]arg list'
		arguments to pass on each gccgo compiler/linker invocation.
	-gcflags '[pattern=]arg list'
		arguments to pass on each go tool compile invocation.
	-installsuffix suffix
		a suffix to use in the name of the package installation directory,
		in order to keep output separate from default builds.
		If using the -race flag, the install suffix is automatically set to race
		or, if set explicitly, has _race appended to it. Likewise for the -msan
		and -asan flags. Using a -buildmode option that requires non-default compile
		flags has a similar effect.
	-ldflags '[pattern=]arg list'
		arguments to pass on each go tool link invocation.
	-linkshared
		build code that will be linked against shared libraries previously
		created with -buildmode=shared.
	-mod mode
		module download mode to use: readonly, vendor, or mod.
		By default, if a vendor directory is present and the go version in go.mod
		is 1.14 or higher, the go command acts as if -mod=vendor were set.
		Otherwise, the go command acts as if -mod=readonly were set.
		See https://golang.org/ref/mod#build-commands for details.
	-modcacherw
		leave newly-created directories in the module cache read-write
		instead of making them read-only.
	-modfile file
		in module aware mode, read (and possibly write) an alternate go.mod
		file instead of the one in the module root directory. A file named
		"go.mod" must still be present in order to determine the module root
		directory, but it is not accessed. When -modfile is specified, an
		alternate go.sum file is also used: its path is derived from the
		-modfile flag by trimming the ".mod" extension and appending ".sum".
	-overlay file
		read a JSON config file that provides an overlay for build operations.
		The file is a JSON struct with a single field, named 'Replace', that
		maps each disk file path (a string) to its backing file path, so that
		a build will run as if the disk file path exists with the contents
		given by the backing file paths, or as if the disk file path does not
		exist if its backing file path is empty. Support for the -overlay flag
		has some limitations: importantly, cgo files included from outside the
		include path must be in the same directory as the Go package they are
		included from, and overlays will not appear when binaries and tests are
		run through go run and go test respectively.
	-pgo file
		specify the file path of a profile for profile-guided optimization (PGO).
		When the special name "auto" is specified, for each main package in the
		build, the go command selects a file named "default.pgo" in the package's
		directory if that file exists, and applies it to the (transitive)
		dependencies of the main package (other packages are not affected).
		Special name "off" turns off PGO. The default is "auto".
	-pkgdir dir
		install and load all packages from dir instead of the usual locations.
		For example, when building with a non-standard configuration,
		use -pkgdir to keep generated packages in a separate location.
	-tags tag,list
		a comma-separated list of additional build tags to consider satisfied
		during the build. For more information about build tags, see
		'go help buildconstraint'. (Earlier versions of Go used a
		space-separated list, and that form is deprecated but still recognized.)
	-trimpath
		remove all file system paths from the resulting executable.
		Instead of absolute file system paths, the recorded file names
		will begin either a module path@version (when using modules),
		or a plain import path (when using the standard library, or GOPATH).
	-toolexec 'cmd args'
		a program to use to invoke toolchain programs like vet and asm.
		For example, instead of running asm, the go command will run
		'cmd args /path/to/asm <arguments for asm>'.
		The TOOLEXEC_IMPORTPATH environment variable will be set,
		matching 'go list -f {{.ImportPath}}' for the package being built.

The -asmflags, -gccgoflags, -gcflags, and -ldflags flags accept a
space-separated list of arguments to pass to an underlying tool
during the build. To embed spaces in an element in the list, surround
it with either single or double quotes. The argument list may be
preceded by a package pattern and an equal sign, which restricts
the use of that argument list to the building of packages matching
that pattern (see 'go help packages' for a description of package
patterns). Without a pattern, the argument list applies only to the
packages named on the command line. The flags may be repeated
with different patterns in order to specify different arguments for
different sets of packages. If a package matches patterns given in
multiple flags, the latest match on the command line wins.
For example, 'go build -gcflags=-S fmt' prints the disassembly
only for package fmt, while 'go build -gcflags=all=-S fmt'
prints the disassembly for fmt and all its dependencies.

For more about specifying packages, see 'go help packages'.
For more about where packages and binaries are installed,
run 'go help gopath'.
For more about calling between Go and C/C++, run 'go help c'.

Note: Build adheres to certain conventions such as those described
by 'go help gopath'. Not all projects can follow these conventions,
however. Installations that have their own conventions or that use
a separate software build system may choose to use lower-level
invocations such as 'go tool compile' and 'go tool link' to avoid
some of the overheads and design decisions of the build tool.

See also: go install, go get, go clean.
	`,
}

const concurrentGCBackendCompilationEnabledByDefault = true

func init() {
	// break init cycle
	CmdBuild.Run = runBuild
	CmdInstall.Run = runInstall

	CmdBuild.Flag.StringVar(&cfg.BuildO, "o", "", "output file or directory")

	AddBuildFlags(CmdBuild, DefaultBuildFlags)
	AddBuildFlags(CmdInstall, DefaultBuildFlags)
	CmdInstall.Flag.BoolVar(&cfg.BuildCheckError, "checkerr", false, "set to true to validate only compilation errors and skip binary generation")
	if cfg.Experiment != nil && cfg.Experiment.CoverageRedesign {
		AddCoverFlags(CmdBuild, nil)
		AddCoverFlags(CmdInstall, nil)
	}
}

// Note that flags consulted by other parts of the code
// (for example, buildV) are in cmd/go/internal/cfg.

var (
	forcedAsmflags   []string // internally-forced flags for cmd/asm
	forcedGcflags    []string // internally-forced flags for cmd/compile
	forcedLdflags    []string // internally-forced flags for cmd/link
	forcedGccgoflags []string // internally-forced flags for gccgo
)

var BuildToolchain toolchain = noToolchain{}
var ldBuildmode string

// buildCompiler implements flag.Var.
// It implements Set by updating both
// BuildToolchain and buildContext.Compiler.
type buildCompiler struct{}

func (c buildCompiler) Set(value string) error {
	switch value {
	case "gc":
		BuildToolchain = gcToolchain{}
	case "gccgo":
		BuildToolchain = gccgoToolchain{}
	default:
		return fmt.Errorf("unknown compiler %q", value)
	}
	cfg.BuildToolchainName = value
	cfg.BuildToolchainCompiler = BuildToolchain.compiler
	cfg.BuildToolchainLinker = BuildToolchain.linker
	cfg.BuildContext.Compiler = value
	return nil
}

func (c buildCompiler) String() string {
	return cfg.BuildContext.Compiler
}

func init() {
	switch build.Default.Compiler {
	case "gc", "gccgo":
		buildCompiler{}.Set(build.Default.Compiler)
	}
}

type BuildFlagMask int

const (
	DefaultBuildFlags BuildFlagMask = 0
	OmitModFlag       BuildFlagMask = 1 << iota
	OmitModCommonFlags
	OmitVFlag
)

// AddBuildFlags adds the flags common to the build, clean, get,
// install, list, run, and test commands.
func AddBuildFlags(cmd *base.Command, mask BuildFlagMask) {
	base.AddBuildFlagsNX(&cmd.Flag)
	base.AddChdirFlag(&cmd.Flag)
	cmd.Flag.BoolVar(&cfg.BuildA, "a", false, "")
	cmd.Flag.IntVar(&cfg.BuildP, "p", cfg.BuildP, "")
	if mask&OmitVFlag == 0 {
		cmd.Flag.BoolVar(&cfg.BuildV, "v", false, "")
	}

	cmd.Flag.Var(&load.BuildAsmflags, "asmflags", "")
	cmd.Flag.Var(buildCompiler{}, "compiler", "")
	cmd.Flag.StringVar(&cfg.BuildBuildmode, "buildmode", "default", "")
	cmd.Flag.Var(&load.BuildGcflags, "gcflags", "")
	cmd.Flag.Var(&load.BuildGccgoflags, "gccgoflags", "")
	if mask&OmitModFlag == 0 {
		base.AddModFlag(&cmd.Flag)
	}
	if mask&OmitModCommonFlags == 0 {
		base.AddModCommonFlags(&cmd.Flag)
	} else {
		// Add the overlay flag even when we don't add the rest of the mod common flags.
		// This only affects 'go get' in GOPATH mode, but add the flag anyway for
		// consistency.
		cmd.Flag.StringVar(&fsys.OverlayFile, "overlay", "", "")
	}
	cmd.Flag.StringVar(&cfg.BuildContext.InstallSuffix, "installsuffix", "", "")
	cmd.Flag.Var(&load.BuildLdflags, "ldflags", "")
	cmd.Flag.BoolVar(&cfg.BuildLinkshared, "linkshared", false, "")
	cmd.Flag.StringVar(&cfg.BuildPGO, "pgo", "auto", "")
	cmd.Flag.StringVar(&cfg.BuildPkgdir, "pkgdir", "", "")
	cmd.Flag.BoolVar(&cfg.BuildRace, "race", false, "")
	cmd.Flag.BoolVar(&cfg.BuildMSan, "msan", false, "")
	cmd.Flag.BoolVar(&cfg.BuildASan, "asan", false, "")
	cmd.Flag.Var((*tagsFlag)(&cfg.BuildContext.BuildTags), "tags", "")
	cmd.Flag.Var((*base.StringsFlag)(&cfg.BuildToolexec), "toolexec", "")
	cmd.Flag.BoolVar(&cfg.BuildTrimpath, "trimpath", false, "")
	cmd.Flag.BoolVar(&cfg.BuildWork, "work", false, "")
	cmd.Flag.Var((*buildvcsFlag)(&cfg.BuildBuildvcs), "buildvcs", "")

	// Undocumented, unstable debugging flags.
	cmd.Flag.StringVar(&cfg.DebugActiongraph, "debug-actiongraph", "", "")
	cmd.Flag.StringVar(&cfg.DebugTrace, "debug-trace", "", "")
	cmd.Flag.StringVar(&cfg.DebugRuntimeTrace, "debug-runtime-trace", "", "")
}

// AddCoverFlags adds coverage-related flags to "cmd". If the
// CoverageRedesign experiment is enabled, we add -cover{mode,pkg} to
// the build command and only -coverprofile to the test command. If
// the CoverageRedesign experiment is disabled, -cover* flags are
// added only to the test command.
func AddCoverFlags(cmd *base.Command, coverProfileFlag *string) {
	addCover := false
	if cfg.Experiment != nil && cfg.Experiment.CoverageRedesign {
		// New coverage enabled: both build and test commands get
		// coverage flags.
		addCover = true
	} else {
		// New coverage disabled: only test command gets cover flags.
		addCover = coverProfileFlag != nil
	}
	if addCover {
		cmd.Flag.BoolVar(&cfg.BuildCover, "cover", false, "")
		cmd.Flag.Var(coverFlag{(*coverModeFlag)(&cfg.BuildCoverMode)}, "covermode", "")
		cmd.Flag.Var(coverFlag{commaListFlag{&cfg.BuildCoverPkg}}, "coverpkg", "")
	}
	if coverProfileFlag != nil {
		cmd.Flag.Var(coverFlag{V: stringFlag{coverProfileFlag}}, "coverprofile", "")
	}
}

// tagsFlag is the implementation of the -tags flag.
type tagsFlag []string

func (v *tagsFlag) Set(s string) error {
	// For compatibility with Go 1.12 and earlier, allow "-tags='a b c'" or even just "-tags='a'".
	if strings.Contains(s, " ") || strings.Contains(s, "'") {
		return (*base.StringsFlag)(v).Set(s)
	}

	// Split on commas, ignore empty strings.
	*v = []string{}
	for _, s := range strings.Split(s, ",") {
		if s != "" {
			*v = append(*v, s)
		}
	}
	return nil
}

func (v *tagsFlag) String() string {
	return "<TagsFlag>"
}

// buildvcsFlag is the implementation of the -buildvcs flag.
type buildvcsFlag string

func (f *buildvcsFlag) IsBoolFlag() bool { return true } // allow -buildvcs (without arguments)

func (f *buildvcsFlag) Set(s string) error {
	// https://go.dev/issue/51748: allow "-buildvcs=auto",
	// in addition to the usual "true" and "false".
	if s == "" || s == "auto" {
		*f = "auto"
		return nil
	}

	b, err := strconv.ParseBool(s)
	if err != nil {
		return errors.New("value is neither 'auto' nor a valid bool")
	}
	*f = (buildvcsFlag)(strconv.FormatBool(b)) // convert to canonical "true" or "false"
	return nil
}

func (f *buildvcsFlag) String() string { return string(*f) }

// fileExtSplit expects a filename and returns the name
// and ext (without the dot). If the file has no
// extension, ext will be empty.
func fileExtSplit(file string) (name, ext string) {
	dotExt := filepath.Ext(file)
	name = file[:len(file)-len(dotExt)]
	if dotExt != "" {
		ext = dotExt[1:]
	}
	return
}

func pkgsMain(pkgs []*load.Package) (res []*load.Package) {
	for _, p := range pkgs {
		if p.Name == "main" {
			res = append(res, p)
		}
	}
	return res
}

func pkgsNotMain(pkgs []*load.Package) (res []*load.Package) {
	for _, p := range pkgs {
		if p.Name != "main" {
			res = append(res, p)
		}
	}
	return res
}

func oneMainPkg(pkgs []*load.Package) []*load.Package {
	if len(pkgs) != 1 || pkgs[0].Name != "main" {
		base.Fatalf("-buildmode=%s requires exactly one main package", cfg.BuildBuildmode)
	}
	return pkgs
}

var pkgsFilter = func(pkgs []*load.Package) []*load.Package { return pkgs }

func runBuild(ctx context.Context, cmd *base.Command, args []string) {
	modload.InitWorkfile()
	BuildInit()
	b := NewBuilder("")
	defer func() {
		if err := b.Close(); err != nil {
			base.Fatal(err)
		}
	}()

	pkgs := load.PackagesAndErrors(ctx, load.PackageOpts{AutoVCS: true}, args)
	load.CheckPackageErrors(pkgs)

	explicitO := len(cfg.BuildO) > 0

	if len(pkgs) == 1 && pkgs[0].Name == "main" && cfg.BuildO == "" {
		cfg.BuildO = pkgs[0].DefaultExecName()
		cfg.BuildO += cfg.ExeSuffix
	}

	// sanity check some often mis-used options
	switch cfg.BuildContext.Compiler {
	case "gccgo":
		if load.BuildGcflags.Present() {
			fmt.Println("go build: when using gccgo toolchain, please pass compiler flags using -gccgoflags, not -gcflags")
		}
		if load.BuildLdflags.Present() {
			fmt.Println("go build: when using gccgo toolchain, please pass linker flags using -gccgoflags, not -ldflags")
		}
	case "gc":
		if load.BuildGccgoflags.Present() {
			fmt.Println("go build: when using gc toolchain, please pass compile flags using -gcflags, and linker flags using -ldflags")
		}
	}

	depMode := ModeBuild

	pkgs = omitTestOnly(pkgsFilter(pkgs))

	// Special case -o /dev/null by not writing at all.
	if base.IsNull(cfg.BuildO) {
		cfg.BuildO = ""
	}

	if cfg.Experiment.CoverageRedesign && cfg.BuildCover {
		load.PrepareForCoverageBuild(pkgs)
	}

	if cfg.BuildO != "" {
		// If the -o name exists and is a directory or
		// ends with a slash or backslash, then
		// write all main packages to that directory.
		// Otherwise require only a single package be built.
		if fi, err := os.Stat(cfg.BuildO); (err == nil && fi.IsDir()) ||
			strings.HasSuffix(cfg.BuildO, "/") ||
			strings.HasSuffix(cfg.BuildO, string(os.PathSeparator)) {
			if !explicitO {
				base.Fatalf("go: build output %q already exists and is a directory", cfg.BuildO)
			}
			a := &Action{Mode: "go build"}
			for _, p := range pkgs {
				if p.Name != "main" {
					continue
				}

				p.Target = filepath.Join(cfg.BuildO, p.DefaultExecName())
				p.Target += cfg.ExeSuffix
				p.Stale = true
				p.StaleReason = "build -o flag in use"
				a.Deps = append(a.Deps, b.AutoAction(ModeInstall, depMode, p))
			}
			if len(a.Deps) == 0 {
				base.Fatalf("go: no main packages to build")
			}
			b.Do(ctx, a)
			return
		}
		if len(pkgs) > 1 {
			base.Fatalf("go: cannot write multiple packages to non-directory %s", cfg.BuildO)
		} else if len(pkgs) == 0 {
			base.Fatalf("no packages to build")
		}
		p := pkgs[0]
		p.Target = cfg.BuildO
		p.Stale = true // must build - not up to date
		p.StaleReason = "build -o flag in use"
		a := b.AutoAction(ModeInstall, depMode, p)
		b.Do(ctx, a)
		return
	}

	a := &Action{Mode: "go build"}
	for _, p := range pkgs {
		a.Deps = append(a.Deps, b.AutoAction(ModeBuild, depMode, p))
	}
	if cfg.BuildBuildmode == "shared" {
		a = b.buildmodeShared(ModeBuild, depMode, args, pkgs, a)
	}
	b.Do(ctx, a)
}

var CmdInstall = &base.Command{
	UsageLine: "go install [build flags] [packages]",
	Short:     "compile and install packages and dependencies",
	Long: `
Install compiles and installs the packages named by the import paths.

Executables are installed in the directory named by the GOBIN environment
variable, which defaults to $GOPATH/bin or $HOME/go/bin if the GOPATH
environment variable is not set. Executables in $GOROOT
are installed in $GOROOT/bin or $GOTOOLDIR instead of $GOBIN.

If the arguments have version suffixes (like @latest or @v1.0.0), "go install"
builds packages in module-aware mode, ignoring the go.mod file in the current
directory or any parent directory, if there is one. This is useful for
installing executables without affecting the dependencies of the main module.
To eliminate ambiguity about which module versions are used in the build, the
arguments must satisfy the following constraints:

- Arguments must be package paths or package patterns (with "..." wildcards).
They must not be standard packages (like fmt), meta-patterns (std, cmd,
all), or relative or absolute file paths.

- All arguments must have the same version suffix. Different queries are not
allowed, even if they refer to the same version.

- All arguments must refer to packages in the same module at the same version.

- Package path arguments must refer to main packages. Pattern arguments
will only match main packages.

- No module is considered the "main" module. If the module containing
packages named on the command line has a go.mod file, it must not contain
directives (replace and exclude) that would cause it to be interpreted
differently than if it were the main module. The module must not require
a higher version of itself.

- Vendor directories are not used in any module. (Vendor directories are not
included in the module zip files downloaded by 'go install'.)

If the arguments don't have version suffixes, "go install" may run in
module-aware mode or GOPATH mode, depending on the GO111MODULE environment
variable and the presence of a go.mod file. See 'go help modules' for details.
If module-aware mode is enabled, "go install" runs in the context of the main
module.

When module-aware mode is disabled, non-main packages are installed in the
directory $GOPATH/pkg/$GOOS_$GOARCH. When module-aware mode is enabled,
non-main packages are built and cached but not installed.

Before Go 1.20, the standard library was installed to
$GOROOT/pkg/$GOOS_$GOARCH.
Starting in Go 1.20, the standard library is built and cached but not installed.
Setting GODEBUG=installgoroot=all restores the use of
$GOROOT/pkg/$GOOS_$GOARCH.

For more about build flags, see 'go help build'.

For more about specifying packages, see 'go help packages'.

See also: go build, go get, go clean.
	`,
}

// libname returns the filename to use for the shared library when using
// -buildmode=shared. The rules we use are:
// Use arguments for special 'meta' packages:
//
//	std --> libstd.so
//	std cmd --> libstd,cmd.so
//
// A single non-meta argument with trailing "/..." is special cased:
//
//	foo/... --> libfoo.so
//	(A relative path like "./..."  expands the "." first)
//
// Use import paths for other cases, changing '/' to '-':
//
//	somelib --> libsubdir-somelib.so
//	./ or ../ --> libsubdir-somelib.so
//	gopkg.in/tomb.v2 -> libgopkg.in-tomb.v2.so
//	a/... b/... ---> liba/c,b/d.so - all matching import paths
//
// Name parts are joined with ','.
func libname(args []string, pkgs []*load.Package) (string, error) {
	var libname string
	appendName := func(arg string) {
		if libname == "" {
			libname = arg
		} else {
			libname += "," + arg
		}
	}
	var haveNonMeta bool
	for _, arg := range args {
		if search.IsMetaPackage(arg) {
			appendName(arg)
		} else {
			haveNonMeta = true
		}
	}
	if len(libname) == 0 { // non-meta packages only. use import paths
		if len(args) == 1 && strings.HasSuffix(args[0], "/...") {
			// Special case of "foo/..." as mentioned above.
			arg := strings.TrimSuffix(args[0], "/...")
			if build.IsLocalImport(arg) {
				cwd, _ := os.Getwd()
				bp, _ := cfg.BuildContext.ImportDir(filepath.Join(cwd, arg), build.FindOnly)
				if bp.ImportPath != "" && bp.ImportPath != "." {
					arg = bp.ImportPath
				}
			}
			appendName(strings.ReplaceAll(arg, "/", "-"))
		} else {
			for _, pkg := range pkgs {
				appendName(strings.ReplaceAll(pkg.ImportPath, "/", "-"))
			}
		}
	} else if haveNonMeta { // have both meta package and a non-meta one
		return "", errors.New("mixing of meta and non-meta packages is not allowed")
	}
	// TODO(mwhudson): Needs to change for platforms that use different naming
	// conventions...
	return "lib" + libname + ".so", nil
}

func runInstall(ctx context.Context, cmd *base.Command, args []string) {
	for _, arg := range args {
		if strings.Contains(arg, "@") && !build.IsLocalImport(arg) && !filepath.IsAbs(arg) {
			installOutsideModule(ctx, args)
			return
		}
	}

	modload.InitWorkfile()
	BuildInit()
	pkgs := load.PackagesAndErrors(ctx, load.PackageOpts{AutoVCS: true}, args)
	if cfg.ModulesEnabled && !modload.HasModRoot() {
		haveErrors := false
		allMissingErrors := true
		for _, pkg := range pkgs {
			if pkg.Error == nil {
				continue
			}
			haveErrors = true
			if missingErr := (*modload.ImportMissingError)(nil); !errors.As(pkg.Error, &missingErr) {
				allMissingErrors = false
				break
			}
		}
		if haveErrors && allMissingErrors {
			latestArgs := make([]string, len(args))
			for i := range args {
				latestArgs[i] = args[i] + "@latest"
			}
			hint := strings.Join(latestArgs, " ")
			base.Fatalf("go: 'go install' requires a version when current directory is not in a module\n\tTry 'go install %s' to install the latest version", hint)
		}
	}
	load.CheckPackageErrors(pkgs)

	if cfg.Experiment.CoverageRedesign && cfg.BuildCover {
		load.PrepareForCoverageBuild(pkgs)
	}

	InstallPackages(ctx, args, pkgs)
}

// omitTestOnly returns pkgs with test-only packages removed.
func omitTestOnly(pkgs []*load.Package) []*load.Package {
	var list []*load.Package
	for _, p := range pkgs {
		if len(p.GoFiles)+len(p.CgoFiles) == 0 && !p.Internal.CmdlinePkgLiteral {
			// Package has no source files,
			// perhaps due to build tags or perhaps due to only having *_test.go files.
			// Also, it is only being processed as the result of a wildcard match
			// like ./..., not because it was listed as a literal path on the command line.
			// Ignore it.
			continue
		}
		list = append(list, p)
	}
	return list
}

func InstallPackages(ctx context.Context, patterns []string, pkgs []*load.Package) {
	ctx, span := trace.StartSpan(ctx, "InstallPackages "+strings.Join(patterns, " "))
	defer span.Done()

	if cfg.GOBIN != "" && !filepath.IsAbs(cfg.GOBIN) {
		base.Fatalf("cannot install, GOBIN must be an absolute path")
	}

	pkgs = omitTestOnly(pkgsFilter(pkgs))
	for _, p := range pkgs {
		if p.Target == "" {
			switch {
			case p.Name != "main" && p.Internal.Local && p.ConflictDir == "":
				// Non-executables outside GOPATH need not have a target:
				// we can use the cache to hold the built package archive for use in future builds.
				// The ones inside GOPATH should have a target (in GOPATH/pkg)
				// or else something is wrong and worth reporting (like a ConflictDir).
			case p.Name != "main" && p.Module != nil:
				// Non-executables have no target (except the cache) when building with modules.
			case p.Name != "main" && p.Standard && p.Internal.Build.PkgObj == "":
				// Most packages in std do not need an installed .a, because they can be
				// rebuilt and used directly from the build cache.
				// A few targets (notably those using cgo) still do need to be installed
				// in case the user's environment lacks a C compiler.
			case p.Internal.GobinSubdir:
				base.Errorf("go: cannot install cross-compiled binaries when GOBIN is set")
			case p.Internal.CmdlineFiles:
				base.Errorf("go: no install location for .go files listed on command line (GOBIN not set)")
			case p.ConflictDir != "":
				base.Errorf("go: no install location for %s: hidden by %s", p.Dir, p.ConflictDir)
			default:
				base.Errorf("go: no install location for directory %s outside GOPATH\n"+
					"\tFor more details see: 'go help gopath'", p.Dir)
			}
		}
	}
	base.ExitIfErrors()

	b := NewBuilder("")
	defer func() {
		if err := b.Close(); err != nil {
			base.Fatal(err)
		}
	}()

	depMode := ModeBuild
	a := &Action{Mode: "go install"}
	var tools []*Action
	for _, p := range pkgs {
		// If p is a tool, delay the installation until the end of the build.
		// This avoids installing assemblers/compilers that are being executed
		// by other steps in the build.
		a1 := b.AutoAction(ModeInstall, depMode, p)
		if load.InstallTargetDir(p) == load.ToTool {
			a.Deps = append(a.Deps, a1.Deps...)
			a1.Deps = append(a1.Deps, a)
			tools = append(tools, a1)
			continue
		}
		a.Deps = append(a.Deps, a1)
	}
	if len(tools) > 0 {
		a = &Action{
			Mode: "go install (tools)",
			Deps: tools,
		}
	}

	if cfg.BuildBuildmode == "shared" {
		// Note: If buildmode=shared then only non-main packages
		// are present in the pkgs list, so all the special case code about
		// tools above did not apply, and a is just a simple Action
		// with a list of Deps, one per package named in pkgs,
		// the same as in runBuild.
		a = b.buildmodeShared(ModeInstall, ModeInstall, patterns, pkgs, a)
	}

	b.Do(ctx, a)
	base.ExitIfErrors()

	// Success. If this command is 'go install' with no arguments
	// and the current directory (the implicit argument) is a command,
	// remove any leftover command binary from a previous 'go build'.
	// The binary is installed; it's not needed here anymore.
	// And worse it might be a stale copy, which you don't want to find
	// instead of the installed one if $PATH contains dot.
	// One way to view this behavior is that it is as if 'go install' first
	// runs 'go build' and the moves the generated file to the install dir.
	// See issue 9645.
	if len(patterns) == 0 && len(pkgs) == 1 && pkgs[0].Name == "main" {
		// Compute file 'go build' would have created.
		// If it exists and is an executable file, remove it.
		targ := pkgs[0].DefaultExecName()
		targ += cfg.ExeSuffix
		if filepath.Join(pkgs[0].Dir, targ) != pkgs[0].Target { // maybe $GOBIN is the current directory
			fi, err := os.Stat(targ)
			if err == nil {
				m := fi.Mode()
				if m.IsRegular() {
					if m&0111 != 0 || cfg.Goos == "windows" { // windows never sets executable bit
						os.Remove(targ)
					}
				}
			}
		}
	}
}

// installOutsideModule implements 'go install pkg@version'. It builds and
// installs one or more main packages in module mode while ignoring any go.mod
// in the current directory or parent directories.
//
// See golang.org/issue/40276 for details and rationale.
func installOutsideModule(ctx context.Context, args []string) {
	modload.ForceUseModules = true
	modload.RootMode = modload.NoRoot
	modload.AllowMissingModuleImports()
	modload.Init()
	BuildInit()

	// Load packages. Ignore non-main packages.
	// Print a warning if an argument contains "..." and matches no main packages.
	// PackagesAndErrors already prints warnings for patterns that don't match any
	// packages, so be careful not to double print.
	// TODO(golang.org/issue/40276): don't report errors loading non-main packages
	// matched by a pattern.
	pkgOpts := load.PackageOpts{MainOnly: true}
	pkgs, err := load.PackagesAndErrorsOutsideModule(ctx, pkgOpts, args)
	if err != nil {
		base.Fatal(err)
	}
	load.CheckPackageErrors(pkgs)
	patterns := make([]string, len(args))
	for i, arg := range args {
		patterns[i] = arg[:strings.Index(arg, "@")]
	}

	// Build and install the packages.
	InstallPackages(ctx, patterns, pkgs)
}

// ExecCmd is the command to use to run user binaries.
// Normally it is empty, meaning run the binaries directly.
// If cross-compiling and running on a remote system or
// simulator, it is typically go_GOOS_GOARCH_exec, with
// the target GOOS and GOARCH substituted.
// The -exec flag overrides these defaults.
var ExecCmd []string

// FindExecCmd derives the value of ExecCmd to use.
// It returns that value and leaves ExecCmd set for direct use.
func FindExecCmd() []string {
	if ExecCmd != nil {
		return ExecCmd
	}
	ExecCmd = []string{} // avoid work the second time
	if cfg.Goos == runtime.GOOS && cfg.Goarch == runtime.GOARCH {
		return ExecCmd
	}
	path, err := cfg.LookPath(fmt.Sprintf("go_%s_%s_exec", cfg.Goos, cfg.Goarch))
	if err == nil {
		ExecCmd = []string{path}
	}
	return ExecCmd
}

// A coverFlag is a flag.Value that also implies -cover.
type coverFlag struct{ V flag.Value }

func (f coverFlag) String() string { return f.V.String() }

func (f coverFlag) Set(value string) error {
	if err := f.V.Set(value); err != nil {
		return err
	}
	cfg.BuildCover = true
	return nil
}

type coverModeFlag string

func (f *coverModeFlag) String() string { return string(*f) }
func (f *coverModeFlag) Set(value string) error {
	switch value {
	case "", "set", "count", "atomic":
		*f = coverModeFlag(value)
		cfg.BuildCoverMode = value
		return nil
	default:
		return errors.New(`valid modes are "set", "count", or "atomic"`)
	}
}

// A commaListFlag is a flag.Value representing a comma-separated list.
type commaListFlag struct{ Vals *[]string }

func (f commaListFlag) String() string { return strings.Join(*f.Vals, ",") }

func (f commaListFlag) Set(value string) error {
	if value == "" {
		*f.Vals = nil
	} else {
		*f.Vals = strings.Split(value, ",")
	}
	return nil
}

// A stringFlag is a flag.Value representing a single string.
type stringFlag struct{ val *string }

func (f stringFlag) String() string { return *f.val }
func (f stringFlag) Set(value string) error {
	*f.val = value
	return nil
}
