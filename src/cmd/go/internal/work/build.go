// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"context"
	"errors"
	"fmt"
	"go/build"
	exec "internal/execabs"
	"internal/goroot"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/load"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modload"
	"cmd/go/internal/search"
	"cmd/go/internal/trace"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
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

When compiling a single main package, build writes
the resulting executable to an output file named after
the first source file ('go build ed.go rx.go' writes 'ed' or 'ed.exe')
or the source code directory ('go build unix/sam' writes 'sam' or 'sam.exe').
The '.exe' suffix is added when writing a Windows executable.

When compiling multiple packages or a single non-main package,
build compiles the packages but discards the resulting object,
serving only as a check that the packages can be built.

The -o flag forces build to write the resulting executable or object
to the named output file or directory, instead of the default behavior described
in the last two paragraphs. If the named output is an existing directory or
ends with a slash or backslash, then any resulting executables
will be written to that directory.

The -i flag installs the packages that are dependencies of the target.
The -i flag is deprecated. Compiled packages are cached automatically.

The build flags are shared by the build, clean, get, install, list, run,
and test commands:

	-a
		force rebuilding of packages that are already up-to-date.
	-n
		print the commands but do not run them.
	-p n
		the number of programs, such as build commands or
		test binaries, that can be run in parallel.
		The default is the number of CPUs available.
	-race
		enable data race detection.
		Supported only on linux/amd64, freebsd/amd64, darwin/amd64, windows/amd64,
		linux/ppc64le and linux/arm64 (only for 48-bit VMA).
	-msan
		enable interoperation with memory sanitizer.
		Supported only on linux/amd64, linux/arm64
		and only with Clang/LLVM as the host C compiler.
		On linux/arm64, pie build mode will be used.
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
		flag. Using a -buildmode option that requires non-default compile flags
		has a similar effect.
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
		has some limitations:importantly, cgo files included from outside the
		include path must be  in the same directory as the Go package they are
		included from, and overlays will not appear when binaries and tests are
		run through go run and go test respectively.
	-pkgdir dir
		install and load all packages from dir instead of the usual locations.
		For example, when building with a non-standard configuration,
		use -pkgdir to keep generated packages in a separate location.
	-tags tag,list
		a comma-separated list of build tags to consider satisfied during the
		build. For more information about build tags, see the description of
		build constraints in the documentation for the go/build package.
		(Earlier versions of Go used a space-separated list, and that form
		is deprecated but still recognized.)
	-trimpath
		remove all file system paths from the resulting executable.
		Instead of absolute file system paths, the recorded file names
		will begin with either "go" (for the standard library),
		or a module path@version (when using modules),
		or a plain import path (when using GOPATH).
	-toolexec 'cmd args'
		a program to use to invoke toolchain programs like vet and asm.
		For example, instead of running asm, the go command will run
		'cmd args /path/to/asm <arguments for asm>'.

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

	CmdBuild.Flag.BoolVar(&cfg.BuildI, "i", false, "")
	CmdBuild.Flag.StringVar(&cfg.BuildO, "o", "", "output file or directory")

	CmdInstall.Flag.BoolVar(&cfg.BuildI, "i", false, "")

	AddBuildFlags(CmdBuild, DefaultBuildFlags)
	AddBuildFlags(CmdInstall, DefaultBuildFlags)
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
	cmd.Flag.StringVar(&cfg.BuildPkgdir, "pkgdir", "", "")
	cmd.Flag.BoolVar(&cfg.BuildRace, "race", false, "")
	cmd.Flag.BoolVar(&cfg.BuildMSan, "msan", false, "")
	cmd.Flag.Var((*tagsFlag)(&cfg.BuildContext.BuildTags), "tags", "")
	cmd.Flag.Var((*base.StringsFlag)(&cfg.BuildToolexec), "toolexec", "")
	cmd.Flag.BoolVar(&cfg.BuildTrimpath, "trimpath", false, "")
	cmd.Flag.BoolVar(&cfg.BuildWork, "work", false, "")

	// Undocumented, unstable debugging flags.
	cmd.Flag.StringVar(&cfg.DebugActiongraph, "debug-actiongraph", "", "")
	cmd.Flag.StringVar(&cfg.DebugTrace, "debug-trace", "", "")
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

var runtimeVersion = runtime.Version()

func runBuild(ctx context.Context, cmd *base.Command, args []string) {
	BuildInit()
	var b Builder
	b.Init()

	pkgs := load.PackagesAndErrors(ctx, args)
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
	if cfg.BuildI {
		depMode = ModeInstall
		fmt.Fprint(os.Stderr, "go build: -i flag is deprecated\n")
	}

	pkgs = omitTestOnly(pkgsFilter(pkgs))

	// Special case -o /dev/null by not writing at all.
	if cfg.BuildO == os.DevNull {
		cfg.BuildO = ""
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
				base.Fatalf("go build: build output %q already exists and is a directory", cfg.BuildO)
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
				base.Fatalf("go build: no main packages to build")
			}
			b.Do(ctx, a)
			return
		}
		if len(pkgs) > 1 {
			base.Fatalf("go build: cannot write multiple packages to non-directory %s", cfg.BuildO)
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

- No module is considered the "main" module. If the module containing
packages named on the command line has a go.mod file, it must not contain
directives (replace and exclude) that would cause it to be interpreted
differently than if it were the main module. The module must not require
a higher version of itself.

- Package path arguments must refer to main packages. Pattern arguments
will only match main packages.

If the arguments don't have version suffixes, "go install" may run in
module-aware mode or GOPATH mode, depending on the GO111MODULE environment
variable and the presence of a go.mod file. See 'go help modules' for details.
If module-aware mode is enabled, "go install" runs in the context of the main
module.

When module-aware mode is disabled, other packages are installed in the
directory $GOPATH/pkg/$GOOS_$GOARCH. When module-aware mode is enabled,
other packages are built and cached but not installed.

The -i flag installs the dependencies of the named packages as well.
The -i flag is deprecated. Compiled packages are cached automatically.

For more about the build flags, see 'go help build'.
For more about specifying packages, see 'go help packages'.

See also: go build, go get, go clean.
	`,
}

// libname returns the filename to use for the shared library when using
// -buildmode=shared. The rules we use are:
// Use arguments for special 'meta' packages:
//	std --> libstd.so
//	std cmd --> libstd,cmd.so
// A single non-meta argument with trailing "/..." is special cased:
//	foo/... --> libfoo.so
//	(A relative path like "./..."  expands the "." first)
// Use import paths for other cases, changing '/' to '-':
//	somelib --> libsubdir-somelib.so
//	./ or ../ --> libsubdir-somelib.so
//	gopkg.in/tomb.v2 -> libgopkg.in-tomb.v2.so
//	a/... b/... ---> liba/c,b/d.so - all matching import paths
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
	// TODO(golang.org/issue/41696): print a deprecation message for the -i flag
	// whenever it's set (or just remove it). For now, we don't print a message
	// if all named packages are in GOROOT. cmd/dist (run by make.bash) uses
	// 'go install -i' when bootstrapping, and we don't want to show deprecation
	// messages in that case.
	for _, arg := range args {
		if strings.Contains(arg, "@") && !build.IsLocalImport(arg) && !filepath.IsAbs(arg) {
			if cfg.BuildI {
				fmt.Fprint(os.Stderr, "go install: -i flag is deprecated\n")
			}
			installOutsideModule(ctx, args)
			return
		}
	}

	BuildInit()
	pkgs := load.PackagesAndErrors(ctx, args)
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
			base.Fatalf("go install: version is required when current directory is not in a module\n\tTry 'go install %s' to install the latest version", hint)
		}
	}
	load.CheckPackageErrors(pkgs)
	if cfg.BuildI {
		allGoroot := true
		for _, pkg := range pkgs {
			if !pkg.Goroot {
				allGoroot = false
				break
			}
		}
		if !allGoroot {
			fmt.Fprint(os.Stderr, "go install: -i flag is deprecated\n")
		}
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
			case p.Standard && p.ImportPath == "unsafe":
				// unsafe is a built-in package, has no target
			case p.Name != "main" && p.Internal.Local && p.ConflictDir == "":
				// Non-executables outside GOPATH need not have a target:
				// we can use the cache to hold the built package archive for use in future builds.
				// The ones inside GOPATH should have a target (in GOPATH/pkg)
				// or else something is wrong and worth reporting (like a ConflictDir).
			case p.Name != "main" && p.Module != nil:
				// Non-executables have no target (except the cache) when building with modules.
			case p.Internal.GobinSubdir:
				base.Errorf("go %s: cannot install cross-compiled binaries when GOBIN is set", cfg.CmdName)
			case p.Internal.CmdlineFiles:
				base.Errorf("go %s: no install location for .go files listed on command line (GOBIN not set)", cfg.CmdName)
			case p.ConflictDir != "":
				base.Errorf("go %s: no install location for %s: hidden by %s", cfg.CmdName, p.Dir, p.ConflictDir)
			default:
				base.Errorf("go %s: no install location for directory %s outside GOPATH\n"+
					"\tFor more details see: 'go help gopath'", cfg.CmdName, p.Dir)
			}
		}
	}
	base.ExitIfErrors()

	var b Builder
	b.Init()
	depMode := ModeBuild
	if cfg.BuildI {
		depMode = ModeInstall
	}
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

	// Check that the arguments satisfy syntactic constraints.
	var version string
	for _, arg := range args {
		if i := strings.Index(arg, "@"); i >= 0 {
			version = arg[i+1:]
			if version == "" {
				base.Fatalf("go install %s: version must not be empty", arg)
			}
			break
		}
	}
	patterns := make([]string, len(args))
	for i, arg := range args {
		if !strings.HasSuffix(arg, "@"+version) {
			base.Errorf("go install %s: all arguments must have the same version (@%s)", arg, version)
			continue
		}
		p := arg[:len(arg)-len(version)-1]
		switch {
		case build.IsLocalImport(p):
			base.Errorf("go install %s: argument must be a package path, not a relative path", arg)
		case filepath.IsAbs(p):
			base.Errorf("go install %s: argument must be a package path, not an absolute path", arg)
		case search.IsMetaPackage(p):
			base.Errorf("go install %s: argument must be a package path, not a meta-package", arg)
		case path.Clean(p) != p:
			base.Errorf("go install %s: argument must be a clean package path", arg)
		case !strings.Contains(p, "...") && search.IsStandardImportPath(p) && goroot.IsStandardPackage(cfg.GOROOT, cfg.BuildContext.Compiler, p):
			base.Errorf("go install %s: argument must not be a package in the standard library", arg)
		default:
			patterns[i] = p
		}
	}
	base.ExitIfErrors()
	BuildInit()

	// Query the module providing the first argument, load its go.mod file, and
	// check that it doesn't contain directives that would cause it to be
	// interpreted differently if it were the main module.
	//
	// If multiple modules match the first argument, accept the longest match
	// (first result). It's possible this module won't provide packages named by
	// later arguments, and other modules would. Let's not try to be too
	// magical though.
	allowed := modload.CheckAllowed
	if modload.IsRevisionQuery(version) {
		// Don't check for retractions if a specific revision is requested.
		allowed = nil
	}
	noneSelected := func(path string) (version string) { return "none" }
	qrs, err := modload.QueryPackages(ctx, patterns[0], version, noneSelected, allowed)
	if err != nil {
		base.Fatalf("go install %s: %v", args[0], err)
	}
	installMod := qrs[0].Mod
	data, err := modfetch.GoMod(installMod.Path, installMod.Version)
	if err != nil {
		base.Fatalf("go install %s: %v", args[0], err)
	}
	f, err := modfile.Parse("go.mod", data, nil)
	if err != nil {
		base.Fatalf("go install %s: %s: %v", args[0], installMod, err)
	}
	directiveFmt := "go install %s: %s\n" +
		"\tThe go.mod file for the module providing named packages contains one or\n" +
		"\tmore %s directives. It must not contain directives that would cause\n" +
		"\tit to be interpreted differently than if it were the main module."
	if len(f.Replace) > 0 {
		base.Fatalf(directiveFmt, args[0], installMod, "replace")
	}
	if len(f.Exclude) > 0 {
		base.Fatalf(directiveFmt, args[0], installMod, "exclude")
	}

	// Since we are in NoRoot mode, the build list initially contains only
	// the dummy command-line-arguments module. Add a requirement on the
	// module that provides the packages named on the command line.
	if err := modload.EditBuildList(ctx, nil, []module.Version{installMod}); err != nil {
		base.Fatalf("go install %s: %v", args[0], err)
	}

	// Load packages for all arguments. Ignore non-main packages.
	// Print a warning if an argument contains "..." and matches no main packages.
	// PackagesAndErrors already prints warnings for patterns that don't match any
	// packages, so be careful not to double print.
	matchers := make([]func(string) bool, len(patterns))
	for i, p := range patterns {
		if strings.Contains(p, "...") {
			matchers[i] = search.MatchPattern(p)
		}
	}

	// TODO(golang.org/issue/40276): don't report errors loading non-main packages
	// matched by a pattern.
	pkgs := load.PackagesAndErrors(ctx, patterns)
	load.CheckPackageErrors(pkgs)
	mainPkgs := make([]*load.Package, 0, len(pkgs))
	mainCount := make([]int, len(patterns))
	nonMainCount := make([]int, len(patterns))
	for _, pkg := range pkgs {
		if pkg.Name == "main" {
			mainPkgs = append(mainPkgs, pkg)
			for i := range patterns {
				if matchers[i] != nil && matchers[i](pkg.ImportPath) {
					mainCount[i]++
				}
			}
		} else {
			for i := range patterns {
				if matchers[i] == nil && patterns[i] == pkg.ImportPath {
					base.Errorf("go install: package %s is not a main package", pkg.ImportPath)
				} else if matchers[i] != nil && matchers[i](pkg.ImportPath) {
					nonMainCount[i]++
				}
			}
		}
	}
	base.ExitIfErrors()
	for i, p := range patterns {
		if matchers[i] != nil && mainCount[i] == 0 && nonMainCount[i] > 0 {
			fmt.Fprintf(os.Stderr, "go: warning: %q matched no main packages\n", p)
		}
	}

	// Check that named packages are all provided by the same module.
	for _, pkg := range mainPkgs {
		if pkg.Module == nil {
			// Packages in std, cmd, and their vendored dependencies
			// don't have this field set.
			base.Errorf("go install: package %s not provided by module %s", pkg.ImportPath, installMod)
		} else if pkg.Module.Path != installMod.Path || pkg.Module.Version != installMod.Version {
			base.Errorf("go install: package %s provided by module %s@%s\n\tAll packages must be provided by the same module (%s).", pkg.ImportPath, pkg.Module.Path, pkg.Module.Version, installMod)
		}
	}
	base.ExitIfErrors()

	// Build and install the packages.
	InstallPackages(ctx, patterns, mainPkgs)
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
	path, err := exec.LookPath(fmt.Sprintf("go_%s_%s_exec", cfg.Goos, cfg.Goarch))
	if err == nil {
		ExecCmd = []string{path}
	}
	return ExecCmd
}
