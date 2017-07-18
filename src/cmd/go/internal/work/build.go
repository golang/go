// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"bufio"
	"bytes"
	"container/heap"
	"debug/elf"
	"errors"
	"flag"
	"fmt"
	"go/build"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"cmd/go/internal/base"
	"cmd/go/internal/buildid"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/str"
)

var CmdBuild = &base.Command{
	UsageLine: "build [-o output] [-i] [build flags] [packages]",
	Short:     "compile packages and dependencies",
	Long: `
Build compiles the packages named by the import paths,
along with their dependencies, but it does not install the results.

If the arguments to build are a list of .go files, build treats
them as a list of source files specifying a single package.

When compiling a single main package, build writes
the resulting executable to an output file named after
the first source file ('go build ed.go rx.go' writes 'ed' or 'ed.exe')
or the source code directory ('go build unix/sam' writes 'sam' or 'sam.exe').
The '.exe' suffix is added when writing a Windows executable.

When compiling multiple packages or a single non-main package,
build compiles the packages but discards the resulting object,
serving only as a check that the packages can be built.

When compiling packages, build ignores files that end in '_test.go'.

The -o flag, only allowed when compiling a single package,
forces build to write the resulting executable or object
to the named output file, instead of the default behavior described
in the last two paragraphs.

The -i flag installs the packages that are dependencies of the target.

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
		Supported only on linux/amd64, freebsd/amd64, darwin/amd64 and windows/amd64.
	-msan
		enable interoperation with memory sanitizer.
		Supported only on linux/amd64,
		and only with Clang/LLVM as the host C compiler.
	-v
		print the names of packages as they are compiled.
	-work
		print the name of the temporary work directory and
		do not delete it when exiting.
	-x
		print the commands.

	-asmflags 'flag list'
		arguments to pass on each go tool asm invocation.
	-buildmode mode
		build mode to use. See 'go help buildmode' for more.
	-compiler name
		name of compiler to use, as in runtime.Compiler (gccgo or gc).
	-gccgoflags 'arg list'
		arguments to pass on each gccgo compiler/linker invocation.
	-gcflags 'arg list'
		arguments to pass on each go tool compile invocation.
	-installsuffix suffix
		a suffix to use in the name of the package installation directory,
		in order to keep output separate from default builds.
		If using the -race flag, the install suffix is automatically set to race
		or, if set explicitly, has _race appended to it. Likewise for the -msan
		flag. Using a -buildmode option that requires non-default compile flags
		has a similar effect.
	-ldflags 'flag list'
		arguments to pass on each go tool link invocation.
	-linkshared
		link against shared libraries previously created with
		-buildmode=shared.
	-pkgdir dir
		install and load all packages from dir instead of the usual locations.
		For example, when building with a non-standard configuration,
		use -pkgdir to keep generated packages in a separate location.
	-tags 'tag list'
		a space-separated list of build tags to consider satisfied during the
		build. For more information about build tags, see the description of
		build constraints in the documentation for the go/build package.
	-toolexec 'cmd args'
		a program to use to invoke toolchain programs like vet and asm.
		For example, instead of running asm, the go command will run
		'cmd args /path/to/asm <arguments for asm>'.

All the flags that take a list of arguments accept a space-separated
list of strings. To embed spaces in an element in the list, surround
it with either single or double quotes.

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
	CmdBuild.Flag.StringVar(&cfg.BuildO, "o", "", "output file")

	AddBuildFlags(CmdBuild)
	AddBuildFlags(CmdInstall)
}

// Note that flags consulted by other parts of the code
// (for example, buildV) are in cmd/go/internal/cfg.

var buildAsmflags []string   // -asmflags flag
var buildGcflags []string    // -gcflags flag
var buildGccgoflags []string // -gccgoflags flag

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

// addBuildFlags adds the flags common to the build, clean, get,
// install, list, run, and test commands.
func AddBuildFlags(cmd *base.Command) {
	cmd.Flag.BoolVar(&cfg.BuildA, "a", false, "")
	cmd.Flag.BoolVar(&cfg.BuildN, "n", false, "")
	cmd.Flag.IntVar(&cfg.BuildP, "p", cfg.BuildP, "")
	cmd.Flag.BoolVar(&cfg.BuildV, "v", false, "")
	cmd.Flag.BoolVar(&cfg.BuildX, "x", false, "")

	cmd.Flag.Var((*base.StringsFlag)(&buildAsmflags), "asmflags", "")
	cmd.Flag.Var(buildCompiler{}, "compiler", "")
	cmd.Flag.StringVar(&cfg.BuildBuildmode, "buildmode", "default", "")
	cmd.Flag.Var((*base.StringsFlag)(&buildGcflags), "gcflags", "")
	cmd.Flag.Var((*base.StringsFlag)(&buildGccgoflags), "gccgoflags", "")
	cmd.Flag.StringVar(&cfg.BuildContext.InstallSuffix, "installsuffix", "", "")
	cmd.Flag.Var((*base.StringsFlag)(&cfg.BuildLdflags), "ldflags", "")
	cmd.Flag.BoolVar(&cfg.BuildLinkshared, "linkshared", false, "")
	cmd.Flag.StringVar(&cfg.BuildPkgdir, "pkgdir", "", "")
	cmd.Flag.BoolVar(&cfg.BuildRace, "race", false, "")
	cmd.Flag.BoolVar(&cfg.BuildMSan, "msan", false, "")
	cmd.Flag.Var((*base.StringsFlag)(&cfg.BuildContext.BuildTags), "tags", "")
	cmd.Flag.Var((*base.StringsFlag)(&cfg.BuildToolexec), "toolexec", "")
	cmd.Flag.BoolVar(&cfg.BuildWork, "work", false, "")
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

func BuildModeInit() {
	gccgo := cfg.BuildToolchainName == "gccgo"
	var codegenArg string
	platform := cfg.Goos + "/" + cfg.Goarch
	switch cfg.BuildBuildmode {
	case "archive":
		pkgsFilter = pkgsNotMain
	case "c-archive":
		pkgsFilter = oneMainPkg
		switch platform {
		case "darwin/arm", "darwin/arm64":
			codegenArg = "-shared"
		default:
			switch cfg.Goos {
			case "dragonfly", "freebsd", "linux", "netbsd", "openbsd", "solaris":
				// Use -shared so that the result is
				// suitable for inclusion in a PIE or
				// shared library.
				codegenArg = "-shared"
			}
		}
		cfg.ExeSuffix = ".a"
		ldBuildmode = "c-archive"
	case "c-shared":
		pkgsFilter = oneMainPkg
		if gccgo {
			codegenArg = "-fPIC"
		} else {
			switch platform {
			case "linux/amd64", "linux/arm", "linux/arm64", "linux/386",
				"android/amd64", "android/arm", "android/arm64", "android/386":
				codegenArg = "-shared"
			case "darwin/amd64", "darwin/386":
			default:
				base.Fatalf("-buildmode=c-shared not supported on %s\n", platform)
			}
		}
		ldBuildmode = "c-shared"
	case "default":
		switch platform {
		case "android/arm", "android/arm64", "android/amd64", "android/386":
			codegenArg = "-shared"
			ldBuildmode = "pie"
		case "darwin/arm", "darwin/arm64":
			codegenArg = "-shared"
			fallthrough
		default:
			ldBuildmode = "exe"
		}
	case "exe":
		pkgsFilter = pkgsMain
		ldBuildmode = "exe"
	case "pie":
		if cfg.BuildRace {
			base.Fatalf("-buildmode=pie not supported when -race is enabled")
		}
		if gccgo {
			base.Fatalf("-buildmode=pie not supported by gccgo")
		} else {
			switch platform {
			case "linux/386", "linux/amd64", "linux/arm", "linux/arm64", "linux/ppc64le", "linux/s390x",
				"android/amd64", "android/arm", "android/arm64", "android/386":
				codegenArg = "-shared"
			default:
				base.Fatalf("-buildmode=pie not supported on %s\n", platform)
			}
		}
		ldBuildmode = "pie"
	case "shared":
		pkgsFilter = pkgsNotMain
		if gccgo {
			codegenArg = "-fPIC"
		} else {
			switch platform {
			case "linux/386", "linux/amd64", "linux/arm", "linux/arm64", "linux/ppc64le", "linux/s390x":
			default:
				base.Fatalf("-buildmode=shared not supported on %s\n", platform)
			}
			codegenArg = "-dynlink"
		}
		if cfg.BuildO != "" {
			base.Fatalf("-buildmode=shared and -o not supported together")
		}
		ldBuildmode = "shared"
	case "plugin":
		pkgsFilter = oneMainPkg
		if gccgo {
			codegenArg = "-fPIC"
		} else {
			switch platform {
			case "linux/amd64", "linux/arm", "linux/arm64", "linux/386", "linux/s390x",
				"android/amd64", "android/arm", "android/arm64", "android/386":
			default:
				base.Fatalf("-buildmode=plugin not supported on %s\n", platform)
			}
			codegenArg = "-dynlink"
		}
		cfg.ExeSuffix = ".so"
		ldBuildmode = "plugin"
	default:
		base.Fatalf("buildmode=%s not supported", cfg.BuildBuildmode)
	}
	if cfg.BuildLinkshared {
		if gccgo {
			codegenArg = "-fPIC"
		} else {
			switch platform {
			case "linux/386", "linux/amd64", "linux/arm", "linux/arm64", "linux/ppc64le", "linux/s390x":
				buildAsmflags = append(buildAsmflags, "-D=GOBUILDMODE_shared=1")
			default:
				base.Fatalf("-linkshared not supported on %s\n", platform)
			}
			codegenArg = "-dynlink"
			// TODO(mwhudson): remove -w when that gets fixed in linker.
			cfg.BuildLdflags = append(cfg.BuildLdflags, "-linkshared", "-w")
		}
	}
	if codegenArg != "" {
		if gccgo {
			buildGccgoflags = append([]string{codegenArg}, buildGccgoflags...)
		} else {
			buildAsmflags = append([]string{codegenArg}, buildAsmflags...)
			buildGcflags = append([]string{codegenArg}, buildGcflags...)
		}
		// Don't alter InstallSuffix when modifying default codegen args.
		if cfg.BuildBuildmode != "default" || cfg.BuildLinkshared {
			if cfg.BuildContext.InstallSuffix != "" {
				cfg.BuildContext.InstallSuffix += "_"
			}
			cfg.BuildContext.InstallSuffix += codegenArg[1:]
		}
	}
	if strings.HasPrefix(runtimeVersion, "go1") && !strings.Contains(os.Args[0], "go_bootstrap") {
		buildGcflags = append(buildGcflags, "-goversion", runtimeVersion)
	}
}

var runtimeVersion = runtime.Version()

func runBuild(cmd *base.Command, args []string) {
	InstrumentInit()
	BuildModeInit()
	var b Builder
	b.Init()

	pkgs := load.PackagesForBuild(args)

	if len(pkgs) == 1 && pkgs[0].Name == "main" && cfg.BuildO == "" {
		_, cfg.BuildO = path.Split(pkgs[0].ImportPath)
		cfg.BuildO += cfg.ExeSuffix
	}

	// Special case -o /dev/null by not writing at all.
	if cfg.BuildO == os.DevNull {
		cfg.BuildO = ""
	}

	// sanity check some often mis-used options
	switch cfg.BuildContext.Compiler {
	case "gccgo":
		if len(buildGcflags) != 0 {
			fmt.Println("go build: when using gccgo toolchain, please pass compiler flags using -gccgoflags, not -gcflags")
		}
		if len(cfg.BuildLdflags) != 0 {
			fmt.Println("go build: when using gccgo toolchain, please pass linker flags using -gccgoflags, not -ldflags")
		}
	case "gc":
		if len(buildGccgoflags) != 0 {
			fmt.Println("go build: when using gc toolchain, please pass compile flags using -gcflags, and linker flags using -ldflags")
		}
	}

	depMode := ModeBuild
	if cfg.BuildI {
		depMode = ModeInstall
	}

	if cfg.BuildO != "" {
		if len(pkgs) > 1 {
			base.Fatalf("go build: cannot use -o with multiple packages")
		} else if len(pkgs) == 0 {
			base.Fatalf("no packages to build")
		}
		p := pkgs[0]
		p.Internal.Target = cfg.BuildO
		p.Stale = true // must build - not up to date
		p.StaleReason = "build -o flag in use"
		a := b.Action(ModeInstall, depMode, p)
		b.Do(a)
		return
	}

	pkgs = pkgsFilter(load.Packages(args))

	var a *Action
	if cfg.BuildBuildmode == "shared" {
		if libName, err := libname(args, pkgs); err != nil {
			base.Fatalf("%s", err.Error())
		} else {
			a = b.libaction(libName, pkgs, ModeBuild, depMode)
		}
	} else {
		a = &Action{}
		for _, p := range pkgs {
			a.Deps = append(a.Deps, b.Action(ModeBuild, depMode, p))
		}
	}
	b.Do(a)
}

var CmdInstall = &base.Command{
	UsageLine: "install [build flags] [packages]",
	Short:     "compile and install packages and dependencies",
	Long: `
Install compiles and installs the packages named by the import paths,
along with their dependencies.

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
		if load.IsMetaPackage(arg) {
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
			appendName(strings.Replace(arg, "/", "-", -1))
		} else {
			for _, pkg := range pkgs {
				appendName(strings.Replace(pkg.ImportPath, "/", "-", -1))
			}
		}
	} else if haveNonMeta { // have both meta package and a non-meta one
		return "", errors.New("mixing of meta and non-meta packages is not allowed")
	}
	// TODO(mwhudson): Needs to change for platforms that use different naming
	// conventions...
	return "lib" + libname + ".so", nil
}

func runInstall(cmd *base.Command, args []string) {
	InstrumentInit()
	BuildModeInit()
	InstallPackages(args, false)
}

func InstallPackages(args []string, forGet bool) {
	if cfg.GOBIN != "" && !filepath.IsAbs(cfg.GOBIN) {
		base.Fatalf("cannot install, GOBIN must be an absolute path")
	}

	pkgs := pkgsFilter(load.PackagesForBuild(args))

	for _, p := range pkgs {
		if p.Target == "" && (!p.Standard || p.ImportPath != "unsafe") {
			switch {
			case p.Internal.GobinSubdir:
				base.Errorf("go install: cannot install cross-compiled binaries when GOBIN is set")
			case p.Internal.Cmdline:
				base.Errorf("go install: no install location for .go files listed on command line (GOBIN not set)")
			case p.ConflictDir != "":
				base.Errorf("go install: no install location for %s: hidden by %s", p.Dir, p.ConflictDir)
			default:
				base.Errorf("go install: no install location for directory %s outside GOPATH\n"+
					"\tFor more details see: 'go help gopath'", p.Dir)
			}
		}
	}
	base.ExitIfErrors()

	var b Builder
	b.Init()
	var a *Action
	if cfg.BuildBuildmode == "shared" {
		if libName, err := libname(args, pkgs); err != nil {
			base.Fatalf("%s", err.Error())
		} else {
			a = b.libaction(libName, pkgs, ModeInstall, ModeInstall)
		}
	} else {
		a = &Action{}
		var tools []*Action
		for _, p := range pkgs {
			// During 'go get', don't attempt (and fail) to install packages with only tests.
			// TODO(rsc): It's not clear why 'go get' should be different from 'go install' here. See #20760.
			if forGet && len(p.GoFiles)+len(p.CgoFiles) == 0 && len(p.TestGoFiles)+len(p.XTestGoFiles) > 0 {
				continue
			}
			// If p is a tool, delay the installation until the end of the build.
			// This avoids installing assemblers/compilers that are being executed
			// by other steps in the build.
			// cmd/cgo is handled specially in b.Action, so that we can
			// both build and use it in the same 'go install'.
			Action := b.Action(ModeInstall, ModeInstall, p)
			if load.GoTools[p.ImportPath] == load.ToTool && p.ImportPath != "cmd/cgo" {
				a.Deps = append(a.Deps, Action.Deps...)
				Action.Deps = append(Action.Deps, a)
				tools = append(tools, Action)
				continue
			}
			a.Deps = append(a.Deps, Action)
		}
		if len(tools) > 0 {
			a = &Action{
				Deps: tools,
			}
		}
	}
	b.Do(a)
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
	if len(args) == 0 && len(pkgs) == 1 && pkgs[0].Name == "main" {
		// Compute file 'go build' would have created.
		// If it exists and is an executable file, remove it.
		_, targ := filepath.Split(pkgs[0].ImportPath)
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

// A Builder holds global state about a build.
// It does not hold per-package state, because we
// build packages in parallel, and the builder is shared.
type Builder struct {
	WorkDir     string               // the temporary work directory (ends in filepath.Separator)
	actionCache map[cacheKey]*Action // a cache of already-constructed actions
	mkdirCache  map[string]bool      // a cache of created directories
	flagCache   map[string]bool      // a cache of supported compiler flags
	Print       func(args ...interface{}) (int, error)

	output    sync.Mutex
	scriptDir string // current directory in printed script

	exec      sync.Mutex
	readySema chan bool
	ready     actionQueue
}

// NOTE: Much of Action would not need to be exported if not for test.
// Maybe test functionality should move into this package too?

// An Action represents a single action in the action graph.
type Action struct {
	Package    *load.Package                 // the package this action works on
	Deps       []*Action                     // actions that must happen before this one
	Func       func(*Builder, *Action) error // the action itself (nil = no-op)
	IgnoreFail bool                          // whether to run f even if dependencies fail
	TestOutput *bytes.Buffer                 // test output buffer
	Args       []string                      // additional args for runProgram

	triggers []*Action // inverse of deps
	cgo      *Action   // action for cgo binary if needed

	// Generated files, directories.
	Link   bool   // target is executable, not just package
	Pkgdir string // the -I or -L argument to use when importing this package
	Objdir string // directory for intermediate objects
	Objpkg string // the intermediate package .a file created during the action
	Target string // goal of the action: the created package or executable

	// Execution state.
	pending  int  // number of deps yet to complete
	priority int  // relative execution priority
	Failed   bool // whether the action failed
}

// cacheKey is the key for the action cache.
type cacheKey struct {
	mode  BuildMode
	p     *load.Package
	shlib string
}

// BuildMode specifies the build mode:
// are we just building things or also installing the results?
type BuildMode int

const (
	ModeBuild BuildMode = iota
	ModeInstall
)

func (b *Builder) Init() {
	var err error
	b.Print = func(a ...interface{}) (int, error) {
		return fmt.Fprint(os.Stderr, a...)
	}
	b.actionCache = make(map[cacheKey]*Action)
	b.mkdirCache = make(map[string]bool)

	if cfg.BuildN {
		b.WorkDir = "$WORK"
	} else {
		b.WorkDir, err = ioutil.TempDir("", "go-build")
		if err != nil {
			base.Fatalf("%s", err)
		}
		if cfg.BuildX || cfg.BuildWork {
			fmt.Fprintf(os.Stderr, "WORK=%s\n", b.WorkDir)
		}
		if !cfg.BuildWork {
			workdir := b.WorkDir
			base.AtExit(func() { os.RemoveAll(workdir) })
		}
	}
}

// readpkglist returns the list of packages that were built into the shared library
// at shlibpath. For the native toolchain this list is stored, newline separated, in
// an ELF note with name "Go\x00\x00" and type 1. For GCCGO it is extracted from the
// .go_export section.
func readpkglist(shlibpath string) (pkgs []*load.Package) {
	var stk load.ImportStack
	if cfg.BuildToolchainName == "gccgo" {
		f, _ := elf.Open(shlibpath)
		sect := f.Section(".go_export")
		data, _ := sect.Data()
		scanner := bufio.NewScanner(bytes.NewBuffer(data))
		for scanner.Scan() {
			t := scanner.Text()
			if strings.HasPrefix(t, "pkgpath ") {
				t = strings.TrimPrefix(t, "pkgpath ")
				t = strings.TrimSuffix(t, ";")
				pkgs = append(pkgs, load.LoadPackage(t, &stk))
			}
		}
	} else {
		pkglistbytes, err := buildid.ReadELFNote(shlibpath, "Go\x00\x00", 1)
		if err != nil {
			base.Fatalf("readELFNote failed: %v", err)
		}
		scanner := bufio.NewScanner(bytes.NewBuffer(pkglistbytes))
		for scanner.Scan() {
			t := scanner.Text()
			pkgs = append(pkgs, load.LoadPackage(t, &stk))
		}
	}
	return
}

// Action returns the action for applying the given operation (mode) to the package.
// depMode is the action to use when building dependencies.
// action never looks for p in a shared library, but may find p's dependencies in a
// shared library if buildLinkshared is true.
func (b *Builder) Action(mode BuildMode, depMode BuildMode, p *load.Package) *Action {
	return b.action1(mode, depMode, p, false, "")
}

// action1 returns the action for applying the given operation (mode) to the package.
// depMode is the action to use when building dependencies.
// action1 will look for p in a shared library if lookshared is true.
// forShlib is the shared library that p will become part of, if any.
func (b *Builder) action1(mode BuildMode, depMode BuildMode, p *load.Package, lookshared bool, forShlib string) *Action {
	shlib := ""
	if lookshared {
		shlib = p.Shlib
	}
	key := cacheKey{mode, p, shlib}

	a := b.actionCache[key]
	if a != nil {
		return a
	}
	if shlib != "" {
		key2 := cacheKey{ModeInstall, nil, shlib}
		a = b.actionCache[key2]
		if a != nil {
			b.actionCache[key] = a
			return a
		}
		pkgs := readpkglist(shlib)
		a = b.libaction(filepath.Base(shlib), pkgs, ModeInstall, depMode)
		b.actionCache[key2] = a
		b.actionCache[key] = a
		return a
	}

	a = &Action{Package: p, Pkgdir: p.Internal.Build.PkgRoot}
	if p.Internal.Pkgdir != "" { // overrides p.t
		a.Pkgdir = p.Internal.Pkgdir
	}
	b.actionCache[key] = a

	for _, p1 := range p.Internal.Imports {
		if forShlib != "" {
			// p is part of a shared library.
			if p1.Shlib != "" && p1.Shlib != forShlib {
				// p1 is explicitly part of a different shared library.
				// Put the action for that shared library into a.Deps.
				a.Deps = append(a.Deps, b.action1(depMode, depMode, p1, true, p1.Shlib))
			} else {
				// p1 is (implicitly or not) part of this shared library.
				// Put the action for p1 into a.Deps.
				a.Deps = append(a.Deps, b.action1(depMode, depMode, p1, false, forShlib))
			}
		} else {
			// p is not part of a shared library.
			// If p1 is in a shared library, put the action for that into
			// a.Deps, otherwise put the action for p1 into a.Deps.
			a.Deps = append(a.Deps, b.action1(depMode, depMode, p1, cfg.BuildLinkshared, p1.Shlib))
		}
	}

	// If we are not doing a cross-build, then record the binary we'll
	// generate for cgo as a dependency of the build of any package
	// using cgo, to make sure we do not overwrite the binary while
	// a package is using it. If this is a cross-build, then the cgo we
	// are writing is not the cgo we need to use.
	if cfg.Goos == runtime.GOOS && cfg.Goarch == runtime.GOARCH && !cfg.BuildRace && !cfg.BuildMSan {
		if (len(p.CgoFiles) > 0 || p.Standard && p.ImportPath == "runtime/cgo") && !cfg.BuildLinkshared && cfg.BuildBuildmode != "shared" {
			var stk load.ImportStack
			p1 := load.LoadPackage("cmd/cgo", &stk)
			if p1.Error != nil {
				base.Fatalf("load cmd/cgo: %v", p1.Error)
			}
			a.cgo = b.Action(depMode, depMode, p1)
			a.Deps = append(a.Deps, a.cgo)
		}
	}

	if p.Standard {
		switch p.ImportPath {
		case "builtin", "unsafe":
			// Fake packages - nothing to build.
			return a
		}
		// gccgo standard library is "fake" too.
		if cfg.BuildToolchainName == "gccgo" {
			// the target name is needed for cgo.
			a.Target = p.Internal.Target
			return a
		}
	}

	if !p.Stale && p.Internal.Target != "" {
		// p.Stale==false implies that p.Internal.Target is up-to-date.
		// Record target name for use by actions depending on this one.
		a.Target = p.Internal.Target
		return a
	}

	if p.Internal.Local && p.Internal.Target == "" {
		// Imported via local path. No permanent target.
		mode = ModeBuild
	}
	work := p.Internal.Pkgdir
	if work == "" {
		work = b.WorkDir
	}
	a.Objdir = filepath.Join(work, a.Package.ImportPath, "_obj") + string(filepath.Separator)
	a.Objpkg = BuildToolchain.Pkgpath(work, a.Package)
	a.Link = p.Name == "main"

	switch mode {
	case ModeInstall:
		a.Func = BuildInstallFunc
		a.Deps = []*Action{b.action1(ModeBuild, depMode, p, lookshared, forShlib)}
		a.Target = a.Package.Internal.Target

		// Install header for cgo in c-archive and c-shared modes.
		if p.UsesCgo() && (cfg.BuildBuildmode == "c-archive" || cfg.BuildBuildmode == "c-shared") {
			hdrTarget := a.Target[:len(a.Target)-len(filepath.Ext(a.Target))] + ".h"
			if cfg.BuildContext.Compiler == "gccgo" {
				// For the header file, remove the "lib"
				// added by go/build, so we generate pkg.h
				// rather than libpkg.h.
				dir, file := filepath.Split(hdrTarget)
				file = strings.TrimPrefix(file, "lib")
				hdrTarget = filepath.Join(dir, file)
			}
			ah := &Action{
				Package: a.Package,
				Deps:    []*Action{a.Deps[0]},
				Func:    (*Builder).installHeader,
				Pkgdir:  a.Pkgdir,
				Objdir:  a.Objdir,
				Target:  hdrTarget,
			}
			a.Deps = append(a.Deps, ah)
		}

	case ModeBuild:
		a.Func = (*Builder).build
		a.Target = a.Objpkg
		if a.Link {
			// An executable file. (This is the name of a temporary file.)
			// Because we run the temporary file in 'go run' and 'go test',
			// the name will show up in ps listings. If the caller has specified
			// a name, use that instead of a.out. The binary is generated
			// in an otherwise empty subdirectory named exe to avoid
			// naming conflicts. The only possible conflict is if we were
			// to create a top-level package named exe.
			name := "a.out"
			if p.Internal.ExeName != "" {
				name = p.Internal.ExeName
			} else if cfg.Goos == "darwin" && cfg.BuildBuildmode == "c-shared" && p.Internal.Target != "" {
				// On OS X, the linker output name gets recorded in the
				// shared library's LC_ID_DYLIB load command.
				// The code invoking the linker knows to pass only the final
				// path element. Arrange that the path element matches what
				// we'll install it as; otherwise the library is only loadable as "a.out".
				_, name = filepath.Split(p.Internal.Target)
			}
			a.Target = a.Objdir + filepath.Join("exe", name) + cfg.ExeSuffix
		}
	}

	return a
}

func (b *Builder) libaction(libname string, pkgs []*load.Package, mode, depMode BuildMode) *Action {
	a := &Action{}
	switch mode {
	default:
		base.Fatalf("unrecognized mode %v", mode)

	case ModeBuild:
		a.Func = (*Builder).linkShared
		a.Target = filepath.Join(b.WorkDir, libname)
		for _, p := range pkgs {
			if p.Internal.Target == "" {
				continue
			}
			a.Deps = append(a.Deps, b.Action(depMode, depMode, p))
		}

	case ModeInstall:
		// Currently build mode shared forces external linking mode, and
		// external linking mode forces an import of runtime/cgo (and
		// math on arm). So if it was not passed on the command line and
		// it is not present in another shared library, add it here.
		gccgo := cfg.BuildToolchainName == "gccgo"
		if !gccgo {
			seencgo := false
			for _, p := range pkgs {
				seencgo = seencgo || (p.Standard && p.ImportPath == "runtime/cgo")
			}
			if !seencgo {
				var stk load.ImportStack
				p := load.LoadPackage("runtime/cgo", &stk)
				if p.Error != nil {
					base.Fatalf("load runtime/cgo: %v", p.Error)
				}
				load.ComputeStale(p)
				// If runtime/cgo is in another shared library, then that's
				// also the shared library that contains runtime, so
				// something will depend on it and so runtime/cgo's staleness
				// will be checked when processing that library.
				if p.Shlib == "" || p.Shlib == libname {
					pkgs = append([]*load.Package{}, pkgs...)
					pkgs = append(pkgs, p)
				}
			}
			if cfg.Goarch == "arm" {
				seenmath := false
				for _, p := range pkgs {
					seenmath = seenmath || (p.Standard && p.ImportPath == "math")
				}
				if !seenmath {
					var stk load.ImportStack
					p := load.LoadPackage("math", &stk)
					if p.Error != nil {
						base.Fatalf("load math: %v", p.Error)
					}
					load.ComputeStale(p)
					// If math is in another shared library, then that's
					// also the shared library that contains runtime, so
					// something will depend on it and so math's staleness
					// will be checked when processing that library.
					if p.Shlib == "" || p.Shlib == libname {
						pkgs = append([]*load.Package{}, pkgs...)
						pkgs = append(pkgs, p)
					}
				}
			}
		}

		// Figure out where the library will go.
		var libdir string
		for _, p := range pkgs {
			plibdir := p.Internal.Build.PkgTargetRoot
			if gccgo {
				plibdir = filepath.Join(plibdir, "shlibs")
			}
			if libdir == "" {
				libdir = plibdir
			} else if libdir != plibdir {
				base.Fatalf("multiple roots %s & %s", libdir, plibdir)
			}
		}
		a.Target = filepath.Join(libdir, libname)

		// Now we can check whether we need to rebuild it.
		stale := false
		var built time.Time
		if fi, err := os.Stat(a.Target); err == nil {
			built = fi.ModTime()
		}
		for _, p := range pkgs {
			if p.Internal.Target == "" {
				continue
			}
			stale = stale || p.Stale
			lstat, err := os.Stat(p.Internal.Target)
			if err != nil || lstat.ModTime().After(built) {
				stale = true
			}
			a.Deps = append(a.Deps, b.action1(depMode, depMode, p, false, a.Target))
		}

		if stale {
			a.Func = BuildInstallFunc
			buildAction := b.libaction(libname, pkgs, ModeBuild, depMode)
			a.Deps = []*Action{buildAction}
			for _, p := range pkgs {
				if p.Internal.Target == "" {
					continue
				}
				shlibnameaction := &Action{}
				shlibnameaction.Func = (*Builder).installShlibname
				shlibnameaction.Target = p.Internal.Target[:len(p.Internal.Target)-2] + ".shlibname"
				a.Deps = append(a.Deps, shlibnameaction)
				shlibnameaction.Deps = append(shlibnameaction.Deps, buildAction)
			}
		}
	}
	return a
}

// ActionList returns the list of actions in the dag rooted at root
// as visited in a depth-first post-order traversal.
func ActionList(root *Action) []*Action {
	seen := map[*Action]bool{}
	all := []*Action{}
	var walk func(*Action)
	walk = func(a *Action) {
		if seen[a] {
			return
		}
		seen[a] = true
		for _, a1 := range a.Deps {
			walk(a1)
		}
		all = append(all, a)
	}
	walk(root)
	return all
}

// allArchiveActions returns a list of the archive dependencies of root.
// This is needed because if package p depends on package q that is in libr.so, the
// action graph looks like p->libr.so->q and so just scanning through p's
// dependencies does not find the import dir for q.
func allArchiveActions(root *Action) []*Action {
	seen := map[*Action]bool{}
	r := []*Action{}
	var walk func(*Action)
	walk = func(a *Action) {
		if seen[a] {
			return
		}
		seen[a] = true
		if strings.HasSuffix(a.Target, ".so") || a == root {
			for _, a1 := range a.Deps {
				walk(a1)
			}
		} else if strings.HasSuffix(a.Target, ".a") {
			r = append(r, a)
		}
	}
	walk(root)
	return r
}

// do runs the action graph rooted at root.
func (b *Builder) Do(root *Action) {
	if _, ok := cfg.OSArchSupportsCgo[cfg.Goos+"/"+cfg.Goarch]; !ok && cfg.BuildContext.Compiler == "gc" {
		fmt.Fprintf(os.Stderr, "cmd/go: unsupported GOOS/GOARCH pair %s/%s\n", cfg.Goos, cfg.Goarch)
		os.Exit(2)
	}
	for _, tag := range cfg.BuildContext.BuildTags {
		if strings.Contains(tag, ",") {
			fmt.Fprintf(os.Stderr, "cmd/go: -tags space-separated list contains comma\n")
			os.Exit(2)
		}
	}

	// Build list of all actions, assigning depth-first post-order priority.
	// The original implementation here was a true queue
	// (using a channel) but it had the effect of getting
	// distracted by low-level leaf actions to the detriment
	// of completing higher-level actions. The order of
	// work does not matter much to overall execution time,
	// but when running "go test std" it is nice to see each test
	// results as soon as possible. The priorities assigned
	// ensure that, all else being equal, the execution prefers
	// to do what it would have done first in a simple depth-first
	// dependency order traversal.
	all := ActionList(root)
	for i, a := range all {
		a.priority = i
	}

	b.readySema = make(chan bool, len(all))

	// Initialize per-action execution state.
	for _, a := range all {
		for _, a1 := range a.Deps {
			a1.triggers = append(a1.triggers, a)
		}
		a.pending = len(a.Deps)
		if a.pending == 0 {
			b.ready.push(a)
			b.readySema <- true
		}
	}

	// Handle runs a single action and takes care of triggering
	// any actions that are runnable as a result.
	handle := func(a *Action) {
		var err error
		if a.Func != nil && (!a.Failed || a.IgnoreFail) {
			err = a.Func(b, a)
		}

		// The actions run in parallel but all the updates to the
		// shared work state are serialized through b.exec.
		b.exec.Lock()
		defer b.exec.Unlock()

		if err != nil {
			if err == errPrintedOutput {
				base.SetExitStatus(2)
			} else {
				base.Errorf("%s", err)
			}
			a.Failed = true
		}

		for _, a0 := range a.triggers {
			if a.Failed {
				a0.Failed = true
			}
			if a0.pending--; a0.pending == 0 {
				b.ready.push(a0)
				b.readySema <- true
			}
		}

		if a == root {
			close(b.readySema)
		}
	}

	var wg sync.WaitGroup

	// Kick off goroutines according to parallelism.
	// If we are using the -n flag (just printing commands)
	// drop the parallelism to 1, both to make the output
	// deterministic and because there is no real work anyway.
	par := cfg.BuildP
	if cfg.BuildN {
		par = 1
	}
	for i := 0; i < par; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case _, ok := <-b.readySema:
					if !ok {
						return
					}
					// Receiving a value from b.readySema entitles
					// us to take from the ready queue.
					b.exec.Lock()
					a := b.ready.pop()
					b.exec.Unlock()
					handle(a)
				case <-base.Interrupted:
					base.SetExitStatus(1)
					return
				}
			}
		}()
	}

	wg.Wait()
}

// build is the action for building a single package or command.
func (b *Builder) build(a *Action) (err error) {
	// Return an error for binary-only package.
	// We only reach this if isStale believes the binary form is
	// either not present or not usable.
	if a.Package.BinaryOnly {
		return fmt.Errorf("missing or invalid package binary for binary-only package %s", a.Package.ImportPath)
	}

	// Return an error if the package has CXX files but it's not using
	// cgo nor SWIG, since the CXX files can only be processed by cgo
	// and SWIG.
	if len(a.Package.CXXFiles) > 0 && !a.Package.UsesCgo() && !a.Package.UsesSwig() {
		return fmt.Errorf("can't build package %s because it contains C++ files (%s) but it's not using cgo nor SWIG",
			a.Package.ImportPath, strings.Join(a.Package.CXXFiles, ","))
	}
	// Same as above for Objective-C files
	if len(a.Package.MFiles) > 0 && !a.Package.UsesCgo() && !a.Package.UsesSwig() {
		return fmt.Errorf("can't build package %s because it contains Objective-C files (%s) but it's not using cgo nor SWIG",
			a.Package.ImportPath, strings.Join(a.Package.MFiles, ","))
	}
	// Same as above for Fortran files
	if len(a.Package.FFiles) > 0 && !a.Package.UsesCgo() && !a.Package.UsesSwig() {
		return fmt.Errorf("can't build package %s because it contains Fortran files (%s) but it's not using cgo nor SWIG",
			a.Package.ImportPath, strings.Join(a.Package.FFiles, ","))
	}

	defer func() {
		if err != nil && err != errPrintedOutput {
			err = fmt.Errorf("go build %s: %v", a.Package.ImportPath, err)
		}
	}()
	if cfg.BuildN {
		// In -n mode, print a banner between packages.
		// The banner is five lines so that when changes to
		// different sections of the bootstrap script have to
		// be merged, the banners give patch something
		// to use to find its context.
		b.Print("\n#\n# " + a.Package.ImportPath + "\n#\n\n")
	}

	if cfg.BuildV {
		b.Print(a.Package.ImportPath + "\n")
	}

	// Make build directory.
	obj := a.Objdir
	if err := b.Mkdir(obj); err != nil {
		return err
	}

	// make target directory
	dir, _ := filepath.Split(a.Target)
	if dir != "" {
		if err := b.Mkdir(dir); err != nil {
			return err
		}
	}

	var gofiles, cgofiles, objdirCgofiles, cfiles, sfiles, cxxfiles, objects, cgoObjects, pcCFLAGS, pcLDFLAGS []string

	gofiles = append(gofiles, a.Package.GoFiles...)
	cgofiles = append(cgofiles, a.Package.CgoFiles...)
	cfiles = append(cfiles, a.Package.CFiles...)
	sfiles = append(sfiles, a.Package.SFiles...)
	cxxfiles = append(cxxfiles, a.Package.CXXFiles...)

	if a.Package.UsesCgo() || a.Package.UsesSwig() {
		if pcCFLAGS, pcLDFLAGS, err = b.getPkgConfigFlags(a.Package); err != nil {
			return
		}
	}

	// Run SWIG on each .swig and .swigcxx file.
	// Each run will generate two files, a .go file and a .c or .cxx file.
	// The .go file will use import "C" and is to be processed by cgo.
	if a.Package.UsesSwig() {
		outGo, outC, outCXX, err := b.swig(a.Package, obj, pcCFLAGS)
		if err != nil {
			return err
		}
		objdirCgofiles = append(objdirCgofiles, outGo...)
		cfiles = append(cfiles, outC...)
		cxxfiles = append(cxxfiles, outCXX...)
	}

	// Run cgo.
	if a.Package.UsesCgo() || a.Package.UsesSwig() {
		// In a package using cgo, cgo compiles the C, C++ and assembly files with gcc.
		// There is one exception: runtime/cgo's job is to bridge the
		// cgo and non-cgo worlds, so it necessarily has files in both.
		// In that case gcc only gets the gcc_* files.
		var gccfiles []string
		gccfiles = append(gccfiles, cfiles...)
		cfiles = nil
		if a.Package.Standard && a.Package.ImportPath == "runtime/cgo" {
			filter := func(files, nongcc, gcc []string) ([]string, []string) {
				for _, f := range files {
					if strings.HasPrefix(f, "gcc_") {
						gcc = append(gcc, f)
					} else {
						nongcc = append(nongcc, f)
					}
				}
				return nongcc, gcc
			}
			sfiles, gccfiles = filter(sfiles, sfiles[:0], gccfiles)
		} else {
			for _, sfile := range sfiles {
				data, err := ioutil.ReadFile(filepath.Join(a.Package.Dir, sfile))
				if err == nil {
					if bytes.HasPrefix(data, []byte("TEXT")) || bytes.Contains(data, []byte("\nTEXT")) ||
						bytes.HasPrefix(data, []byte("DATA")) || bytes.Contains(data, []byte("\nDATA")) ||
						bytes.HasPrefix(data, []byte("GLOBL")) || bytes.Contains(data, []byte("\nGLOBL")) {
						return fmt.Errorf("package using cgo has Go assembly file %s", sfile)
					}
				}
			}
			gccfiles = append(gccfiles, sfiles...)
			sfiles = nil
		}

		var cgoExe string
		if a.cgo != nil && a.cgo.Target != "" {
			cgoExe = a.cgo.Target
		} else {
			cgoExe = base.Tool("cgo")
		}
		outGo, outObj, err := b.cgo(a, cgoExe, obj, pcCFLAGS, pcLDFLAGS, cgofiles, objdirCgofiles, gccfiles, cxxfiles, a.Package.MFiles, a.Package.FFiles)
		if err != nil {
			return err
		}
		if cfg.BuildToolchainName == "gccgo" {
			cgoObjects = append(cgoObjects, filepath.Join(a.Objdir, "_cgo_flags"))
		}
		cgoObjects = append(cgoObjects, outObj...)
		gofiles = append(gofiles, outGo...)
	}

	if len(gofiles) == 0 {
		return &load.NoGoError{Package: a.Package}
	}

	// If we're doing coverage, preprocess the .go files and put them in the work directory
	if a.Package.Internal.CoverMode != "" {
		for i, file := range gofiles {
			var sourceFile string
			var coverFile string
			var key string
			if strings.HasSuffix(file, ".cgo1.go") {
				// cgo files have absolute paths
				base := filepath.Base(file)
				sourceFile = file
				coverFile = filepath.Join(obj, base)
				key = strings.TrimSuffix(base, ".cgo1.go") + ".go"
			} else {
				sourceFile = filepath.Join(a.Package.Dir, file)
				coverFile = filepath.Join(obj, file)
				key = file
			}
			cover := a.Package.Internal.CoverVars[key]
			if cover == nil || base.IsTestFile(file) {
				// Not covering this file.
				continue
			}
			if err := b.cover(a, coverFile, sourceFile, 0666, cover.Var); err != nil {
				return err
			}
			gofiles[i] = coverFile
		}
	}

	// Prepare Go import path list.
	inc := b.includeArgs("-I", allArchiveActions(a))

	// Compile Go.
	ofile, out, err := BuildToolchain.gc(b, a.Package, a.Objpkg, obj, len(sfiles) > 0, inc, gofiles)
	if len(out) > 0 {
		b.showOutput(a.Package.Dir, a.Package.ImportPath, b.processOutput(out))
		if err != nil {
			return errPrintedOutput
		}
	}
	if err != nil {
		return err
	}
	if ofile != a.Objpkg {
		objects = append(objects, ofile)
	}

	// Copy .h files named for goos or goarch or goos_goarch
	// to names using GOOS and GOARCH.
	// For example, defs_linux_amd64.h becomes defs_GOOS_GOARCH.h.
	_goos_goarch := "_" + cfg.Goos + "_" + cfg.Goarch
	_goos := "_" + cfg.Goos
	_goarch := "_" + cfg.Goarch
	for _, file := range a.Package.HFiles {
		name, ext := fileExtSplit(file)
		switch {
		case strings.HasSuffix(name, _goos_goarch):
			targ := file[:len(name)-len(_goos_goarch)] + "_GOOS_GOARCH." + ext
			if err := b.copyFile(a, obj+targ, filepath.Join(a.Package.Dir, file), 0666, true); err != nil {
				return err
			}
		case strings.HasSuffix(name, _goarch):
			targ := file[:len(name)-len(_goarch)] + "_GOARCH." + ext
			if err := b.copyFile(a, obj+targ, filepath.Join(a.Package.Dir, file), 0666, true); err != nil {
				return err
			}
		case strings.HasSuffix(name, _goos):
			targ := file[:len(name)-len(_goos)] + "_GOOS." + ext
			if err := b.copyFile(a, obj+targ, filepath.Join(a.Package.Dir, file), 0666, true); err != nil {
				return err
			}
		}
	}

	for _, file := range cfiles {
		out := file[:len(file)-len(".c")] + ".o"
		if err := BuildToolchain.cc(b, a.Package, obj, obj+out, file); err != nil {
			return err
		}
		objects = append(objects, out)
	}

	// Assemble .s files.
	if len(sfiles) > 0 {
		ofiles, err := BuildToolchain.asm(b, a.Package, obj, sfiles)
		if err != nil {
			return err
		}
		objects = append(objects, ofiles...)
	}

	// NOTE(rsc): On Windows, it is critically important that the
	// gcc-compiled objects (cgoObjects) be listed after the ordinary
	// objects in the archive. I do not know why this is.
	// https://golang.org/issue/2601
	objects = append(objects, cgoObjects...)

	// Add system object files.
	for _, syso := range a.Package.SysoFiles {
		objects = append(objects, filepath.Join(a.Package.Dir, syso))
	}

	// Pack into archive in obj directory.
	// If the Go compiler wrote an archive, we only need to add the
	// object files for non-Go sources to the archive.
	// If the Go compiler wrote an archive and the package is entirely
	// Go sources, there is no pack to execute at all.
	if len(objects) > 0 {
		if err := BuildToolchain.pack(b, a.Package, obj, a.Objpkg, objects); err != nil {
			return err
		}
	}

	// Link if needed.
	if a.Link {
		// The compiler only cares about direct imports, but the
		// linker needs the whole dependency tree.
		all := ActionList(a)
		all = all[:len(all)-1] // drop a
		if err := BuildToolchain.ld(b, a, a.Target, all, a.Objpkg, objects); err != nil {
			return err
		}
	}

	return nil
}

// PkgconfigCmd returns a pkg-config binary name
// defaultPkgConfig is defined in zdefaultcc.go, written by cmd/dist.
func (b *Builder) PkgconfigCmd() string {
	return envList("PKG_CONFIG", cfg.DefaultPkgConfig)[0]
}

// splitPkgConfigOutput parses the pkg-config output into a slice of
// flags. pkg-config always uses \ to escape special characters.
func splitPkgConfigOutput(out []byte) []string {
	if len(out) == 0 {
		return nil
	}
	var flags []string
	flag := make([]byte, len(out))
	r, w := 0, 0
	for r < len(out) {
		switch out[r] {
		case ' ', '\t', '\r', '\n':
			if w > 0 {
				flags = append(flags, string(flag[:w]))
			}
			w = 0
		case '\\':
			r++
			fallthrough
		default:
			if r < len(out) {
				flag[w] = out[r]
				w++
			}
		}
		r++
	}
	if w > 0 {
		flags = append(flags, string(flag[:w]))
	}
	return flags
}

// Calls pkg-config if needed and returns the cflags/ldflags needed to build the package.
func (b *Builder) getPkgConfigFlags(p *load.Package) (cflags, ldflags []string, err error) {
	if pkgs := p.CgoPkgConfig; len(pkgs) > 0 {
		var out []byte
		out, err = b.runOut(p.Dir, p.ImportPath, nil, b.PkgconfigCmd(), "--cflags", pkgs)
		if err != nil {
			b.showOutput(p.Dir, b.PkgconfigCmd()+" --cflags "+strings.Join(pkgs, " "), string(out))
			b.Print(err.Error() + "\n")
			err = errPrintedOutput
			return
		}
		if len(out) > 0 {
			cflags = splitPkgConfigOutput(out)
		}
		out, err = b.runOut(p.Dir, p.ImportPath, nil, b.PkgconfigCmd(), "--libs", pkgs)
		if err != nil {
			b.showOutput(p.Dir, b.PkgconfigCmd()+" --libs "+strings.Join(pkgs, " "), string(out))
			b.Print(err.Error() + "\n")
			err = errPrintedOutput
			return
		}
		if len(out) > 0 {
			ldflags = strings.Fields(string(out))
		}
	}
	return
}

func (b *Builder) installShlibname(a *Action) error {
	a1 := a.Deps[0]
	err := ioutil.WriteFile(a.Target, []byte(filepath.Base(a1.Target)+"\n"), 0666)
	if err != nil {
		return err
	}
	if cfg.BuildX {
		b.Showcmd("", "echo '%s' > %s # internal", filepath.Base(a1.Target), a.Target)
	}
	return nil
}

func (b *Builder) linkShared(a *Action) (err error) {
	allactions := ActionList(a)
	allactions = allactions[:len(allactions)-1]
	return BuildToolchain.ldShared(b, a.Deps, a.Target, allactions)
}

// BuildInstallFunc is the action for installing a single package or executable.
func BuildInstallFunc(b *Builder, a *Action) (err error) {
	defer func() {
		if err != nil && err != errPrintedOutput {
			err = fmt.Errorf("go install %s: %v", a.Package.ImportPath, err)
		}
	}()
	a1 := a.Deps[0]
	perm := os.FileMode(0666)
	if a1.Link {
		switch cfg.BuildBuildmode {
		case "c-archive", "c-shared", "plugin":
		default:
			perm = 0777
		}
	}

	// make target directory
	dir, _ := filepath.Split(a.Target)
	if dir != "" {
		if err := b.Mkdir(dir); err != nil {
			return err
		}
	}

	// remove object dir to keep the amount of
	// garbage down in a large build. On an operating system
	// with aggressive buffering, cleaning incrementally like
	// this keeps the intermediate objects from hitting the disk.
	if !cfg.BuildWork {
		defer os.RemoveAll(a1.Objdir)
		defer os.Remove(a1.Target)
	}

	return b.moveOrCopyFile(a, a.Target, a1.Target, perm, false)
}

// includeArgs returns the -I or -L directory list for access
// to the results of the list of actions.
func (b *Builder) includeArgs(flag string, all []*Action) []string {
	inc := []string{}
	incMap := map[string]bool{
		b.WorkDir:     true, // handled later
		cfg.GOROOTpkg: true,
		"":            true, // ignore empty strings
	}

	// Look in the temporary space for results of test-specific actions.
	// This is the $WORK/my/package/_test directory for the
	// package being built, so there are few of these.
	for _, a1 := range all {
		if a1.Package == nil {
			continue
		}
		if dir := a1.Pkgdir; dir != a1.Package.Internal.Build.PkgRoot && !incMap[dir] {
			incMap[dir] = true
			inc = append(inc, flag, dir)
		}
	}

	// Also look in $WORK for any non-test packages that have
	// been built but not installed.
	inc = append(inc, flag, b.WorkDir)

	// Finally, look in the installed package directories for each action.
	// First add the package dirs corresponding to GOPATH entries
	// in the original GOPATH order.
	need := map[string]*build.Package{}
	for _, a1 := range all {
		if a1.Package != nil && a1.Pkgdir == a1.Package.Internal.Build.PkgRoot {
			need[a1.Package.Internal.Build.Root] = a1.Package.Internal.Build
		}
	}
	for _, root := range cfg.Gopath {
		if p := need[root]; p != nil && !incMap[p.PkgRoot] {
			incMap[p.PkgRoot] = true
			inc = append(inc, flag, p.PkgTargetRoot)
		}
	}

	// Then add anything that's left.
	for _, a1 := range all {
		if a1.Package == nil {
			continue
		}
		if dir := a1.Pkgdir; dir == a1.Package.Internal.Build.PkgRoot && !incMap[dir] {
			incMap[dir] = true
			inc = append(inc, flag, a1.Package.Internal.Build.PkgTargetRoot)
		}
	}

	return inc
}

// moveOrCopyFile is like 'mv src dst' or 'cp src dst'.
func (b *Builder) moveOrCopyFile(a *Action, dst, src string, perm os.FileMode, force bool) error {
	if cfg.BuildN {
		b.Showcmd("", "mv %s %s", src, dst)
		return nil
	}

	// If we can update the mode and rename to the dst, do it.
	// Otherwise fall back to standard copy.

	// If the destination directory has the group sticky bit set,
	// we have to copy the file to retain the correct permissions.
	// https://golang.org/issue/18878
	if fi, err := os.Stat(filepath.Dir(dst)); err == nil {
		if fi.IsDir() && (fi.Mode()&os.ModeSetgid) != 0 {
			return b.copyFile(a, dst, src, perm, force)
		}
	}

	// The perm argument is meant to be adjusted according to umask,
	// but we don't know what the umask is.
	// Create a dummy file to find out.
	// This avoids build tags and works even on systems like Plan 9
	// where the file mask computation incorporates other information.
	mode := perm
	f, err := os.OpenFile(filepath.Clean(dst)+"-go-tmp-umask", os.O_WRONLY|os.O_CREATE|os.O_EXCL, perm)
	if err == nil {
		fi, err := f.Stat()
		if err == nil {
			mode = fi.Mode() & 0777
		}
		name := f.Name()
		f.Close()
		os.Remove(name)
	}

	if err := os.Chmod(src, mode); err == nil {
		if err := os.Rename(src, dst); err == nil {
			if cfg.BuildX {
				b.Showcmd("", "mv %s %s", src, dst)
			}
			return nil
		}
	}

	return b.copyFile(a, dst, src, perm, force)
}

// copyFile is like 'cp src dst'.
func (b *Builder) copyFile(a *Action, dst, src string, perm os.FileMode, force bool) error {
	if cfg.BuildN || cfg.BuildX {
		b.Showcmd("", "cp %s %s", src, dst)
		if cfg.BuildN {
			return nil
		}
	}

	sf, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sf.Close()

	// Be careful about removing/overwriting dst.
	// Do not remove/overwrite if dst exists and is a directory
	// or a non-object file.
	if fi, err := os.Stat(dst); err == nil {
		if fi.IsDir() {
			return fmt.Errorf("build output %q already exists and is a directory", dst)
		}
		if !force && fi.Mode().IsRegular() && !isObject(dst) {
			return fmt.Errorf("build output %q already exists and is not an object file", dst)
		}
	}

	// On Windows, remove lingering ~ file from last attempt.
	if base.ToolIsWindows {
		if _, err := os.Stat(dst + "~"); err == nil {
			os.Remove(dst + "~")
		}
	}

	mayberemovefile(dst)
	df, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	if err != nil && base.ToolIsWindows {
		// Windows does not allow deletion of a binary file
		// while it is executing. Try to move it out of the way.
		// If the move fails, which is likely, we'll try again the
		// next time we do an install of this binary.
		if err := os.Rename(dst, dst+"~"); err == nil {
			os.Remove(dst + "~")
		}
		df, err = os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	}
	if err != nil {
		return err
	}

	_, err = io.Copy(df, sf)
	df.Close()
	if err != nil {
		mayberemovefile(dst)
		return fmt.Errorf("copying %s to %s: %v", src, dst, err)
	}
	return nil
}

// Install the cgo export header file, if there is one.
func (b *Builder) installHeader(a *Action) error {
	src := a.Objdir + "_cgo_install.h"
	if _, err := os.Stat(src); os.IsNotExist(err) {
		// If the file does not exist, there are no exported
		// functions, and we do not install anything.
		return nil
	}

	dir, _ := filepath.Split(a.Target)
	if dir != "" {
		if err := b.Mkdir(dir); err != nil {
			return err
		}
	}

	return b.moveOrCopyFile(a, a.Target, src, 0666, true)
}

// cover runs, in effect,
//	go tool cover -mode=b.coverMode -var="varName" -o dst.go src.go
func (b *Builder) cover(a *Action, dst, src string, perm os.FileMode, varName string) error {
	return b.run(a.Objdir, "cover "+a.Package.ImportPath, nil,
		cfg.BuildToolexec,
		base.Tool("cover"),
		"-mode", a.Package.Internal.CoverMode,
		"-var", varName,
		"-o", dst,
		src)
}

var objectMagic = [][]byte{
	{'!', '<', 'a', 'r', 'c', 'h', '>', '\n'}, // Package archive
	{'\x7F', 'E', 'L', 'F'},                   // ELF
	{0xFE, 0xED, 0xFA, 0xCE},                  // Mach-O big-endian 32-bit
	{0xFE, 0xED, 0xFA, 0xCF},                  // Mach-O big-endian 64-bit
	{0xCE, 0xFA, 0xED, 0xFE},                  // Mach-O little-endian 32-bit
	{0xCF, 0xFA, 0xED, 0xFE},                  // Mach-O little-endian 64-bit
	{0x4d, 0x5a, 0x90, 0x00, 0x03, 0x00},      // PE (Windows) as generated by 6l/8l and gcc
	{0x00, 0x00, 0x01, 0xEB},                  // Plan 9 i386
	{0x00, 0x00, 0x8a, 0x97},                  // Plan 9 amd64
	{0x00, 0x00, 0x06, 0x47},                  // Plan 9 arm
}

func isObject(s string) bool {
	f, err := os.Open(s)
	if err != nil {
		return false
	}
	defer f.Close()
	buf := make([]byte, 64)
	io.ReadFull(f, buf)
	for _, magic := range objectMagic {
		if bytes.HasPrefix(buf, magic) {
			return true
		}
	}
	return false
}

// mayberemovefile removes a file only if it is a regular file
// When running as a user with sufficient privileges, we may delete
// even device files, for example, which is not intended.
func mayberemovefile(s string) {
	if fi, err := os.Lstat(s); err == nil && !fi.Mode().IsRegular() {
		return
	}
	os.Remove(s)
}

// fmtcmd formats a command in the manner of fmt.Sprintf but also:
//
//	If dir is non-empty and the script is not in dir right now,
//	fmtcmd inserts "cd dir\n" before the command.
//
//	fmtcmd replaces the value of b.WorkDir with $WORK.
//	fmtcmd replaces the value of goroot with $GOROOT.
//	fmtcmd replaces the value of b.gobin with $GOBIN.
//
//	fmtcmd replaces the name of the current directory with dot (.)
//	but only when it is at the beginning of a space-separated token.
//
func (b *Builder) fmtcmd(dir string, format string, args ...interface{}) string {
	cmd := fmt.Sprintf(format, args...)
	if dir != "" && dir != "/" {
		cmd = strings.Replace(" "+cmd, " "+dir, " .", -1)[1:]
		if b.scriptDir != dir {
			b.scriptDir = dir
			cmd = "cd " + dir + "\n" + cmd
		}
	}
	if b.WorkDir != "" {
		cmd = strings.Replace(cmd, b.WorkDir, "$WORK", -1)
	}
	return cmd
}

// showcmd prints the given command to standard output
// for the implementation of -n or -x.
func (b *Builder) Showcmd(dir string, format string, args ...interface{}) {
	b.output.Lock()
	defer b.output.Unlock()
	b.Print(b.fmtcmd(dir, format, args...) + "\n")
}

// showOutput prints "# desc" followed by the given output.
// The output is expected to contain references to 'dir', usually
// the source directory for the package that has failed to build.
// showOutput rewrites mentions of dir with a relative path to dir
// when the relative path is shorter. This is usually more pleasant.
// For example, if fmt doesn't compile and we are in src/html,
// the output is
//
//	$ go build
//	# fmt
//	../fmt/print.go:1090: undefined: asdf
//	$
//
// instead of
//
//	$ go build
//	# fmt
//	/usr/gopher/go/src/fmt/print.go:1090: undefined: asdf
//	$
//
// showOutput also replaces references to the work directory with $WORK.
//
func (b *Builder) showOutput(dir, desc, out string) {
	prefix := "# " + desc
	suffix := "\n" + out
	if reldir := base.ShortPath(dir); reldir != dir {
		suffix = strings.Replace(suffix, " "+dir, " "+reldir, -1)
		suffix = strings.Replace(suffix, "\n"+dir, "\n"+reldir, -1)
	}
	suffix = strings.Replace(suffix, " "+b.WorkDir, " $WORK", -1)

	b.output.Lock()
	defer b.output.Unlock()
	b.Print(prefix, suffix)
}

// errPrintedOutput is a special error indicating that a command failed
// but that it generated output as well, and that output has already
// been printed, so there's no point showing 'exit status 1' or whatever
// the wait status was. The main executor, builder.do, knows not to
// print this error.
var errPrintedOutput = errors.New("already printed output - no need to show error")

var cgoLine = regexp.MustCompile(`\[[^\[\]]+\.cgo1\.go:[0-9]+(:[0-9]+)?\]`)
var cgoTypeSigRe = regexp.MustCompile(`\b_Ctype_\B`)

// run runs the command given by cmdline in the directory dir.
// If the command fails, run prints information about the failure
// and returns a non-nil error.
func (b *Builder) run(dir string, desc string, env []string, cmdargs ...interface{}) error {
	out, err := b.runOut(dir, desc, env, cmdargs...)
	if len(out) > 0 {
		if desc == "" {
			desc = b.fmtcmd(dir, "%s", strings.Join(str.StringList(cmdargs...), " "))
		}
		b.showOutput(dir, desc, b.processOutput(out))
		if err != nil {
			err = errPrintedOutput
		}
	}
	return err
}

// processOutput prepares the output of runOut to be output to the console.
func (b *Builder) processOutput(out []byte) string {
	if out[len(out)-1] != '\n' {
		out = append(out, '\n')
	}
	messages := string(out)
	// Fix up output referring to cgo-generated code to be more readable.
	// Replace x.go:19[/tmp/.../x.cgo1.go:18] with x.go:19.
	// Replace *[100]_Ctype_foo with *[100]C.foo.
	// If we're using -x, assume we're debugging and want the full dump, so disable the rewrite.
	if !cfg.BuildX && cgoLine.MatchString(messages) {
		messages = cgoLine.ReplaceAllString(messages, "")
		messages = cgoTypeSigRe.ReplaceAllString(messages, "C.")
	}
	return messages
}

// runOut runs the command given by cmdline in the directory dir.
// It returns the command output and any errors that occurred.
func (b *Builder) runOut(dir string, desc string, env []string, cmdargs ...interface{}) ([]byte, error) {
	cmdline := str.StringList(cmdargs...)
	if cfg.BuildN || cfg.BuildX {
		var envcmdline string
		for i := range env {
			envcmdline += env[i]
			envcmdline += " "
		}
		envcmdline += joinUnambiguously(cmdline)
		b.Showcmd(dir, "%s", envcmdline)
		if cfg.BuildN {
			return nil, nil
		}
	}

	nbusy := 0
	for {
		var buf bytes.Buffer
		cmd := exec.Command(cmdline[0], cmdline[1:]...)
		cmd.Stdout = &buf
		cmd.Stderr = &buf
		cmd.Dir = dir
		cmd.Env = base.MergeEnvLists(env, base.EnvForDir(cmd.Dir, os.Environ()))
		err := cmd.Run()

		// cmd.Run will fail on Unix if some other process has the binary
		// we want to run open for writing. This can happen here because
		// we build and install the cgo command and then run it.
		// If another command was kicked off while we were writing the
		// cgo binary, the child process for that command may be holding
		// a reference to the fd, keeping us from running exec.
		//
		// But, you might reasonably wonder, how can this happen?
		// The cgo fd, like all our fds, is close-on-exec, so that we need
		// not worry about other processes inheriting the fd accidentally.
		// The answer is that running a command is fork and exec.
		// A child forked while the cgo fd is open inherits that fd.
		// Until the child has called exec, it holds the fd open and the
		// kernel will not let us run cgo. Even if the child were to close
		// the fd explicitly, it would still be open from the time of the fork
		// until the time of the explicit close, and the race would remain.
		//
		// On Unix systems, this results in ETXTBSY, which formats
		// as "text file busy". Rather than hard-code specific error cases,
		// we just look for that string. If this happens, sleep a little
		// and try again. We let this happen three times, with increasing
		// sleep lengths: 100+200+400 ms = 0.7 seconds.
		//
		// An alternate solution might be to split the cmd.Run into
		// separate cmd.Start and cmd.Wait, and then use an RWLock
		// to make sure that copyFile only executes when no cmd.Start
		// call is in progress. However, cmd.Start (really syscall.forkExec)
		// only guarantees that when it returns, the exec is committed to
		// happen and succeed. It uses a close-on-exec file descriptor
		// itself to determine this, so we know that when cmd.Start returns,
		// at least one close-on-exec file descriptor has been closed.
		// However, we cannot be sure that all of them have been closed,
		// so the program might still encounter ETXTBSY even with such
		// an RWLock. The race window would be smaller, perhaps, but not
		// guaranteed to be gone.
		//
		// Sleeping when we observe the race seems to be the most reliable
		// option we have.
		//
		// https://golang.org/issue/3001
		//
		if err != nil && nbusy < 3 && strings.Contains(err.Error(), "text file busy") {
			time.Sleep(100 * time.Millisecond << uint(nbusy))
			nbusy++
			continue
		}

		// err can be something like 'exit status 1'.
		// Add information about what program was running.
		// Note that if buf.Bytes() is non-empty, the caller usually
		// shows buf.Bytes() and does not print err at all, so the
		// prefix here does not make most output any more verbose.
		if err != nil {
			err = errors.New(cmdline[0] + ": " + err.Error())
		}
		return buf.Bytes(), err
	}
}

// joinUnambiguously prints the slice, quoting where necessary to make the
// output unambiguous.
// TODO: See issue 5279. The printing of commands needs a complete redo.
func joinUnambiguously(a []string) string {
	var buf bytes.Buffer
	for i, s := range a {
		if i > 0 {
			buf.WriteByte(' ')
		}
		q := strconv.Quote(s)
		if s == "" || strings.Contains(s, " ") || len(q) > len(s)+2 {
			buf.WriteString(q)
		} else {
			buf.WriteString(s)
		}
	}
	return buf.String()
}

// mkdir makes the named directory.
func (b *Builder) Mkdir(dir string) error {
	b.exec.Lock()
	defer b.exec.Unlock()
	// We can be a little aggressive about being
	// sure directories exist. Skip repeated calls.
	if b.mkdirCache[dir] {
		return nil
	}
	b.mkdirCache[dir] = true

	if cfg.BuildN || cfg.BuildX {
		b.Showcmd("", "mkdir -p %s", dir)
		if cfg.BuildN {
			return nil
		}
	}

	if err := os.MkdirAll(dir, 0777); err != nil {
		return err
	}
	return nil
}

// mkAbs returns an absolute path corresponding to
// evaluating f in the directory dir.
// We always pass absolute paths of source files so that
// the error messages will include the full path to a file
// in need of attention.
func mkAbs(dir, f string) string {
	// Leave absolute paths alone.
	// Also, during -n mode we use the pseudo-directory $WORK
	// instead of creating an actual work directory that won't be used.
	// Leave paths beginning with $WORK alone too.
	if filepath.IsAbs(f) || strings.HasPrefix(f, "$WORK") {
		return f
	}
	return filepath.Join(dir, f)
}

type toolchain interface {
	// gc runs the compiler in a specific directory on a set of files
	// and returns the name of the generated output file.
	gc(b *Builder, p *load.Package, archive, obj string, asmhdr bool, importArgs []string, gofiles []string) (ofile string, out []byte, err error)
	// cc runs the toolchain's C compiler in a directory on a C file
	// to produce an output file.
	cc(b *Builder, p *load.Package, objdir, ofile, cfile string) error
	// asm runs the assembler in a specific directory on specific files
	// and returns a list of named output files.
	asm(b *Builder, p *load.Package, obj string, sfiles []string) ([]string, error)
	// pkgpath builds an appropriate path for a temporary package file.
	Pkgpath(basedir string, p *load.Package) string
	// pack runs the archive packer in a specific directory to create
	// an archive from a set of object files.
	// typically it is run in the object directory.
	pack(b *Builder, p *load.Package, objDir, afile string, ofiles []string) error
	// ld runs the linker to create an executable starting at mainpkg.
	ld(b *Builder, root *Action, out string, allactions []*Action, mainpkg string, ofiles []string) error
	// ldShared runs the linker to create a shared library containing the pkgs built by toplevelactions
	ldShared(b *Builder, toplevelactions []*Action, out string, allactions []*Action) error

	compiler() string
	linker() string
}

type noToolchain struct{}

func noCompiler() error {
	log.Fatalf("unknown compiler %q", cfg.BuildContext.Compiler)
	return nil
}

func (noToolchain) compiler() string {
	noCompiler()
	return ""
}

func (noToolchain) linker() string {
	noCompiler()
	return ""
}

func (noToolchain) gc(b *Builder, p *load.Package, archive, obj string, asmhdr bool, importArgs []string, gofiles []string) (ofile string, out []byte, err error) {
	return "", nil, noCompiler()
}

func (noToolchain) asm(b *Builder, p *load.Package, obj string, sfiles []string) ([]string, error) {
	return nil, noCompiler()
}

func (noToolchain) Pkgpath(basedir string, p *load.Package) string {
	noCompiler()
	return ""
}

func (noToolchain) pack(b *Builder, p *load.Package, objDir, afile string, ofiles []string) error {
	return noCompiler()
}

func (noToolchain) ld(b *Builder, root *Action, out string, allactions []*Action, mainpkg string, ofiles []string) error {
	return noCompiler()
}

func (noToolchain) ldShared(b *Builder, toplevelactions []*Action, out string, allactions []*Action) error {
	return noCompiler()
}

func (noToolchain) cc(b *Builder, p *load.Package, objdir, ofile, cfile string) error {
	return noCompiler()
}

// The Go toolchain.
type gcToolchain struct{}

func (gcToolchain) compiler() string {
	return base.Tool("compile")
}

func (gcToolchain) linker() string {
	return base.Tool("link")
}

func (gcToolchain) gc(b *Builder, p *load.Package, archive, obj string, asmhdr bool, importArgs []string, gofiles []string) (ofile string, output []byte, err error) {
	if archive != "" {
		ofile = archive
	} else {
		out := "_go_.o"
		ofile = obj + out
	}

	gcargs := []string{"-p", p.ImportPath}
	if p.Name == "main" {
		gcargs[1] = "main"
	}
	if p.Standard {
		gcargs = append(gcargs, "-std")
	}
	compilingRuntime := p.Standard && (p.ImportPath == "runtime" || strings.HasPrefix(p.ImportPath, "runtime/internal"))
	if compilingRuntime {
		// runtime compiles with a special gc flag to emit
		// additional reflect type data.
		gcargs = append(gcargs, "-+")
	}

	// If we're giving the compiler the entire package (no C etc files), tell it that,
	// so that it can give good error messages about forward declarations.
	// Exceptions: a few standard packages have forward declarations for
	// pieces supplied behind-the-scenes by package runtime.
	extFiles := len(p.CgoFiles) + len(p.CFiles) + len(p.CXXFiles) + len(p.MFiles) + len(p.FFiles) + len(p.SFiles) + len(p.SysoFiles) + len(p.SwigFiles) + len(p.SwigCXXFiles)
	if p.Standard {
		switch p.ImportPath {
		case "bytes", "internal/poll", "net", "os", "runtime/pprof", "sync", "syscall", "time":
			extFiles++
		}
	}
	if extFiles == 0 {
		gcargs = append(gcargs, "-complete")
	}
	if cfg.BuildContext.InstallSuffix != "" {
		gcargs = append(gcargs, "-installsuffix", cfg.BuildContext.InstallSuffix)
	}
	if p.Internal.BuildID != "" {
		gcargs = append(gcargs, "-buildid", p.Internal.BuildID)
	}
	platform := cfg.Goos + "/" + cfg.Goarch
	if p.Internal.OmitDebug || platform == "nacl/amd64p32" || platform == "darwin/arm" || platform == "darwin/arm64" || cfg.Goos == "plan9" {
		gcargs = append(gcargs, "-dwarf=false")
	}

	for _, path := range p.Imports {
		if i := strings.LastIndex(path, "/vendor/"); i >= 0 {
			gcargs = append(gcargs, "-importmap", path[i+len("/vendor/"):]+"="+path)
		} else if strings.HasPrefix(path, "vendor/") {
			gcargs = append(gcargs, "-importmap", path[len("vendor/"):]+"="+path)
		}
	}

	gcflags := buildGcflags
	if compilingRuntime {
		// Remove -N, if present.
		// It is not possible to build the runtime with no optimizations,
		// because the compiler cannot eliminate enough write barriers.
		gcflags = make([]string, len(buildGcflags))
		copy(gcflags, buildGcflags)
		for i := 0; i < len(gcflags); i++ {
			if gcflags[i] == "-N" {
				copy(gcflags[i:], gcflags[i+1:])
				gcflags = gcflags[:len(gcflags)-1]
				i--
			}
		}
	}
	args := []interface{}{cfg.BuildToolexec, base.Tool("compile"), "-o", ofile, "-trimpath", b.WorkDir, gcflags, gcargs, "-D", p.Internal.LocalPrefix, importArgs}
	if ofile == archive {
		args = append(args, "-pack")
	}
	if asmhdr {
		args = append(args, "-asmhdr", obj+"go_asm.h")
	}

	// Add -c=N to use concurrent backend compilation, if possible.
	if c := gcBackendConcurrency(gcflags); c > 1 {
		args = append(args, fmt.Sprintf("-c=%d", c))
	}

	for _, f := range gofiles {
		args = append(args, mkAbs(p.Dir, f))
	}

	output, err = b.runOut(p.Dir, p.ImportPath, nil, args...)
	return ofile, output, err
}

// gcBackendConcurrency returns the backend compiler concurrency level for a package compilation.
func gcBackendConcurrency(gcflags []string) int {
	// First, check whether we can use -c at all for this compilation.
	canDashC := concurrentGCBackendCompilationEnabledByDefault

	switch e := os.Getenv("GO19CONCURRENTCOMPILATION"); e {
	case "0":
		canDashC = false
	case "1":
		canDashC = true
	case "":
		// Not set. Use default.
	default:
		log.Fatalf("GO19CONCURRENTCOMPILATION must be 0, 1, or unset, got %q", e)
	}

	if os.Getenv("GOEXPERIMENT") != "" {
		// Concurrent compilation is presumed incompatible with GOEXPERIMENTs.
		canDashC = false
	}

CheckFlags:
	for _, flag := range gcflags {
		// Concurrent compilation is presumed incompatible with any gcflags,
		// except for a small whitelist of commonly used flags.
		// If the user knows better, they can manually add their own -c to the gcflags.
		switch flag {
		case "-N", "-l", "-S", "-B", "-C", "-I":
			// OK
		default:
			canDashC = false
			break CheckFlags
		}
	}

	if !canDashC {
		return 1
	}

	// Decide how many concurrent backend compilations to allow.
	//
	// If we allow too many, in theory we might end up with p concurrent processes,
	// each with c concurrent backend compiles, all fighting over the same resources.
	// However, in practice, that seems not to happen too much.
	// Most build graphs are surprisingly serial, so p==1 for much of the build.
	// Furthermore, concurrent backend compilation is only enabled for a part
	// of the overall compiler execution, so c==1 for much of the build.
	// So don't worry too much about that interaction for now.
	//
	// However, in practice, setting c above 4 tends not to help very much.
	// See the analysis in CL 41192.
	//
	// TODO(josharian): attempt to detect whether this particular compilation
	// is likely to be a bottleneck, e.g. when:
	//   - it has no successor packages to compile (usually package main)
	//   - all paths through the build graph pass through it
	//   - critical path scheduling says it is high priority
	// and in such a case, set c to runtime.NumCPU.
	// We do this now when p==1.
	if cfg.BuildP == 1 {
		// No process parallelism. Max out c.
		return runtime.NumCPU()
	}
	// Some process parallelism. Set c to min(4, numcpu).
	c := 4
	if ncpu := runtime.NumCPU(); ncpu < c {
		c = ncpu
	}
	return c
}

func (gcToolchain) asm(b *Builder, p *load.Package, obj string, sfiles []string) ([]string, error) {
	// Add -I pkg/GOOS_GOARCH so #include "textflag.h" works in .s files.
	inc := filepath.Join(cfg.GOROOT, "pkg", "include")
	args := []interface{}{cfg.BuildToolexec, base.Tool("asm"), "-trimpath", b.WorkDir, "-I", obj, "-I", inc, "-D", "GOOS_" + cfg.Goos, "-D", "GOARCH_" + cfg.Goarch, buildAsmflags}
	if p.ImportPath == "runtime" && cfg.Goarch == "386" {
		for _, arg := range buildAsmflags {
			if arg == "-dynlink" {
				args = append(args, "-D=GOBUILDMODE_shared=1")
			}
		}
	}
	var ofiles []string
	for _, sfile := range sfiles {
		ofile := obj + sfile[:len(sfile)-len(".s")] + ".o"
		ofiles = append(ofiles, ofile)
		a := append(args, "-o", ofile, mkAbs(p.Dir, sfile))
		if err := b.run(p.Dir, p.ImportPath, nil, a...); err != nil {
			return nil, err
		}
	}
	return ofiles, nil
}

// toolVerify checks that the command line args writes the same output file
// if run using newTool instead.
// Unused now but kept around for future use.
func toolVerify(b *Builder, p *load.Package, newTool string, ofile string, args []interface{}) error {
	newArgs := make([]interface{}, len(args))
	copy(newArgs, args)
	newArgs[1] = base.Tool(newTool)
	newArgs[3] = ofile + ".new" // x.6 becomes x.6.new
	if err := b.run(p.Dir, p.ImportPath, nil, newArgs...); err != nil {
		return err
	}
	data1, err := ioutil.ReadFile(ofile)
	if err != nil {
		return err
	}
	data2, err := ioutil.ReadFile(ofile + ".new")
	if err != nil {
		return err
	}
	if !bytes.Equal(data1, data2) {
		return fmt.Errorf("%s and %s produced different output files:\n%s\n%s", filepath.Base(args[1].(string)), newTool, strings.Join(str.StringList(args...), " "), strings.Join(str.StringList(newArgs...), " "))
	}
	os.Remove(ofile + ".new")
	return nil
}

func (gcToolchain) Pkgpath(basedir string, p *load.Package) string {
	end := filepath.FromSlash(p.ImportPath + ".a")
	return filepath.Join(basedir, end)
}

func (gcToolchain) pack(b *Builder, p *load.Package, objDir, afile string, ofiles []string) error {
	var absOfiles []string
	for _, f := range ofiles {
		absOfiles = append(absOfiles, mkAbs(objDir, f))
	}
	absAfile := mkAbs(objDir, afile)

	// The archive file should have been created by the compiler.
	// Since it used to not work that way, verify.
	if !cfg.BuildN {
		if _, err := os.Stat(absAfile); err != nil {
			base.Fatalf("os.Stat of archive file failed: %v", err)
		}
	}

	if cfg.BuildN || cfg.BuildX {
		cmdline := str.StringList("pack", "r", absAfile, absOfiles)
		b.Showcmd(p.Dir, "%s # internal", joinUnambiguously(cmdline))
	}
	if cfg.BuildN {
		return nil
	}
	if err := packInternal(b, absAfile, absOfiles); err != nil {
		b.showOutput(p.Dir, p.ImportPath, err.Error()+"\n")
		return errPrintedOutput
	}
	return nil
}

func packInternal(b *Builder, afile string, ofiles []string) error {
	dst, err := os.OpenFile(afile, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		return err
	}
	defer dst.Close() // only for error returns or panics
	w := bufio.NewWriter(dst)

	for _, ofile := range ofiles {
		src, err := os.Open(ofile)
		if err != nil {
			return err
		}
		fi, err := src.Stat()
		if err != nil {
			src.Close()
			return err
		}
		// Note: Not using %-16.16s format because we care
		// about bytes, not runes.
		name := fi.Name()
		if len(name) > 16 {
			name = name[:16]
		} else {
			name += strings.Repeat(" ", 16-len(name))
		}
		size := fi.Size()
		fmt.Fprintf(w, "%s%-12d%-6d%-6d%-8o%-10d`\n",
			name, 0, 0, 0, 0644, size)
		n, err := io.Copy(w, src)
		src.Close()
		if err == nil && n < size {
			err = io.ErrUnexpectedEOF
		} else if err == nil && n > size {
			err = fmt.Errorf("file larger than size reported by stat")
		}
		if err != nil {
			return fmt.Errorf("copying %s to %s: %v", ofile, afile, err)
		}
		if size&1 != 0 {
			w.WriteByte(0)
		}
	}

	if err := w.Flush(); err != nil {
		return err
	}
	return dst.Close()
}

// setextld sets the appropriate linker flags for the specified compiler.
func setextld(ldflags []string, compiler []string) []string {
	for _, f := range ldflags {
		if f == "-extld" || strings.HasPrefix(f, "-extld=") {
			// don't override -extld if supplied
			return ldflags
		}
	}
	ldflags = append(ldflags, "-extld="+compiler[0])
	if len(compiler) > 1 {
		extldflags := false
		add := strings.Join(compiler[1:], " ")
		for i, f := range ldflags {
			if f == "-extldflags" && i+1 < len(ldflags) {
				ldflags[i+1] = add + " " + ldflags[i+1]
				extldflags = true
				break
			} else if strings.HasPrefix(f, "-extldflags=") {
				ldflags[i] = "-extldflags=" + add + " " + ldflags[i][len("-extldflags="):]
				extldflags = true
				break
			}
		}
		if !extldflags {
			ldflags = append(ldflags, "-extldflags="+add)
		}
	}
	return ldflags
}

func (gcToolchain) ld(b *Builder, root *Action, out string, allactions []*Action, mainpkg string, ofiles []string) error {
	importArgs := b.includeArgs("-L", allactions)
	cxx := len(root.Package.CXXFiles) > 0 || len(root.Package.SwigCXXFiles) > 0
	for _, a := range allactions {
		if a.Package != nil && (len(a.Package.CXXFiles) > 0 || len(a.Package.SwigCXXFiles) > 0) {
			cxx = true
		}
	}
	var ldflags []string
	if cfg.BuildContext.InstallSuffix != "" {
		ldflags = append(ldflags, "-installsuffix", cfg.BuildContext.InstallSuffix)
	}
	if root.Package.Internal.OmitDebug {
		ldflags = append(ldflags, "-s", "-w")
	}
	if cfg.BuildBuildmode == "plugin" {
		pluginpath := root.Package.ImportPath
		if pluginpath == "command-line-arguments" {
			pluginpath = "plugin/unnamed-" + root.Package.Internal.BuildID
		}
		ldflags = append(ldflags, "-pluginpath", pluginpath)
	}

	// If the user has not specified the -extld option, then specify the
	// appropriate linker. In case of C++ code, use the compiler named
	// by the CXX environment variable or defaultCXX if CXX is not set.
	// Else, use the CC environment variable and defaultCC as fallback.
	var compiler []string
	if cxx {
		compiler = envList("CXX", cfg.DefaultCXX)
	} else {
		compiler = envList("CC", cfg.DefaultCC)
	}
	ldflags = setextld(ldflags, compiler)
	ldflags = append(ldflags, "-buildmode="+ldBuildmode)
	if root.Package.Internal.BuildID != "" {
		ldflags = append(ldflags, "-buildid="+root.Package.Internal.BuildID)
	}
	ldflags = append(ldflags, cfg.BuildLdflags...)

	// On OS X when using external linking to build a shared library,
	// the argument passed here to -o ends up recorded in the final
	// shared library in the LC_ID_DYLIB load command.
	// To avoid putting the temporary output directory name there
	// (and making the resulting shared library useless),
	// run the link in the output directory so that -o can name
	// just the final path element.
	dir := "."
	if cfg.Goos == "darwin" && cfg.BuildBuildmode == "c-shared" {
		dir, out = filepath.Split(out)
	}

	return b.run(dir, root.Package.ImportPath, nil, cfg.BuildToolexec, base.Tool("link"), "-o", out, importArgs, ldflags, mainpkg)
}

func (gcToolchain) ldShared(b *Builder, toplevelactions []*Action, out string, allactions []*Action) error {
	importArgs := b.includeArgs("-L", allactions)
	ldflags := []string{"-installsuffix", cfg.BuildContext.InstallSuffix}
	ldflags = append(ldflags, "-buildmode=shared")
	ldflags = append(ldflags, cfg.BuildLdflags...)
	cxx := false
	for _, a := range allactions {
		if a.Package != nil && (len(a.Package.CXXFiles) > 0 || len(a.Package.SwigCXXFiles) > 0) {
			cxx = true
		}
	}
	// If the user has not specified the -extld option, then specify the
	// appropriate linker. In case of C++ code, use the compiler named
	// by the CXX environment variable or defaultCXX if CXX is not set.
	// Else, use the CC environment variable and defaultCC as fallback.
	var compiler []string
	if cxx {
		compiler = envList("CXX", cfg.DefaultCXX)
	} else {
		compiler = envList("CC", cfg.DefaultCC)
	}
	ldflags = setextld(ldflags, compiler)
	for _, d := range toplevelactions {
		if !strings.HasSuffix(d.Target, ".a") { // omit unsafe etc and actions for other shared libraries
			continue
		}
		ldflags = append(ldflags, d.Package.ImportPath+"="+d.Target)
	}
	return b.run(".", out, nil, cfg.BuildToolexec, base.Tool("link"), "-o", out, importArgs, ldflags)
}

func (gcToolchain) cc(b *Builder, p *load.Package, objdir, ofile, cfile string) error {
	return fmt.Errorf("%s: C source files not supported without cgo", mkAbs(p.Dir, cfile))
}

// The Gccgo toolchain.
type gccgoToolchain struct{}

var GccgoName, GccgoBin string
var gccgoErr error

func init() {
	GccgoName = os.Getenv("GCCGO")
	if GccgoName == "" {
		GccgoName = "gccgo"
	}
	GccgoBin, gccgoErr = exec.LookPath(GccgoName)
}

func (gccgoToolchain) compiler() string {
	checkGccgoBin()
	return GccgoBin
}

func (gccgoToolchain) linker() string {
	checkGccgoBin()
	return GccgoBin
}

func checkGccgoBin() {
	if gccgoErr == nil {
		return
	}
	fmt.Fprintf(os.Stderr, "cmd/go: gccgo: %s\n", gccgoErr)
	os.Exit(2)
}

func (tools gccgoToolchain) gc(b *Builder, p *load.Package, archive, obj string, asmhdr bool, importArgs []string, gofiles []string) (ofile string, output []byte, err error) {
	out := "_go_.o"
	ofile = obj + out
	gcargs := []string{"-g"}
	gcargs = append(gcargs, b.gccArchArgs()...)
	if pkgpath := gccgoPkgpath(p); pkgpath != "" {
		gcargs = append(gcargs, "-fgo-pkgpath="+pkgpath)
	}
	if p.Internal.LocalPrefix != "" {
		gcargs = append(gcargs, "-fgo-relative-import-path="+p.Internal.LocalPrefix)
	}

	// Handle vendor directories
	savedirs := []string{}
	for _, incdir := range importArgs {
		if incdir != "-I" {
			savedirs = append(savedirs, incdir)
		}
	}

	for _, path := range p.Imports {
		// If this is a new vendor path, add it to the list of importArgs
		if i := strings.LastIndex(path, "/vendor"); i >= 0 {
			for _, dir := range savedirs {
				// Check if the vendor path is already included in dir
				if strings.HasSuffix(dir, path[:i+len("/vendor")]) {
					continue
				}
				// Make sure this vendor path is not already in the list for importArgs
				vendorPath := dir + "/" + path[:i+len("/vendor")]
				for _, imp := range importArgs {
					if imp == "-I" {
						continue
					}
					// This vendorPath is already in the list
					if imp == vendorPath {
						goto nextSuffixPath
					}
				}
				// New vendorPath not yet in the importArgs list, so add it
				importArgs = append(importArgs, "-I", vendorPath)
			nextSuffixPath:
			}
		} else if strings.HasPrefix(path, "vendor/") {
			for _, dir := range savedirs {
				// Make sure this vendor path is not already in the list for importArgs
				vendorPath := dir + "/" + path[len("/vendor"):]
				for _, imp := range importArgs {
					if imp == "-I" {
						continue
					}
					if imp == vendorPath {
						goto nextPrefixPath
					}
				}
				// This vendor path is needed and not already in the list, so add it
				importArgs = append(importArgs, "-I", vendorPath)
			nextPrefixPath:
			}
		}
	}

	args := str.StringList(tools.compiler(), importArgs, "-c", gcargs, "-o", ofile, buildGccgoflags)
	for _, f := range gofiles {
		args = append(args, mkAbs(p.Dir, f))
	}

	output, err = b.runOut(p.Dir, p.ImportPath, nil, args)
	return ofile, output, err
}

func (tools gccgoToolchain) asm(b *Builder, p *load.Package, obj string, sfiles []string) ([]string, error) {
	var ofiles []string
	for _, sfile := range sfiles {
		ofile := obj + sfile[:len(sfile)-len(".s")] + ".o"
		ofiles = append(ofiles, ofile)
		sfile = mkAbs(p.Dir, sfile)
		defs := []string{"-D", "GOOS_" + cfg.Goos, "-D", "GOARCH_" + cfg.Goarch}
		if pkgpath := gccgoCleanPkgpath(p); pkgpath != "" {
			defs = append(defs, `-D`, `GOPKGPATH=`+pkgpath)
		}
		defs = tools.maybePIC(defs)
		defs = append(defs, b.gccArchArgs()...)
		err := b.run(p.Dir, p.ImportPath, nil, tools.compiler(), "-xassembler-with-cpp", "-I", obj, "-c", "-o", ofile, defs, sfile)
		if err != nil {
			return nil, err
		}
	}
	return ofiles, nil
}

func (gccgoToolchain) Pkgpath(basedir string, p *load.Package) string {
	end := filepath.FromSlash(p.ImportPath + ".a")
	afile := filepath.Join(basedir, end)
	// add "lib" to the final element
	return filepath.Join(filepath.Dir(afile), "lib"+filepath.Base(afile))
}

func (gccgoToolchain) pack(b *Builder, p *load.Package, objDir, afile string, ofiles []string) error {
	var absOfiles []string
	for _, f := range ofiles {
		absOfiles = append(absOfiles, mkAbs(objDir, f))
	}
	return b.run(p.Dir, p.ImportPath, nil, "ar", "rc", mkAbs(objDir, afile), absOfiles)
}

func (tools gccgoToolchain) link(b *Builder, root *Action, out string, allactions []*Action, mainpkg string, ofiles []string, buildmode, desc string) error {
	// gccgo needs explicit linking with all package dependencies,
	// and all LDFLAGS from cgo dependencies.
	apackagePathsSeen := make(map[string]bool)
	afiles := []string{}
	shlibs := []string{}
	ldflags := b.gccArchArgs()
	cgoldflags := []string{}
	usesCgo := false
	cxx := false
	objc := false
	fortran := false
	if root.Package != nil {
		cxx = len(root.Package.CXXFiles) > 0 || len(root.Package.SwigCXXFiles) > 0
		objc = len(root.Package.MFiles) > 0
		fortran = len(root.Package.FFiles) > 0
	}

	readCgoFlags := func(flagsFile string) error {
		flags, err := ioutil.ReadFile(flagsFile)
		if err != nil {
			return err
		}
		const ldflagsPrefix = "_CGO_LDFLAGS="
		for _, line := range strings.Split(string(flags), "\n") {
			if strings.HasPrefix(line, ldflagsPrefix) {
				newFlags := strings.Fields(line[len(ldflagsPrefix):])
				for _, flag := range newFlags {
					// Every _cgo_flags file has -g and -O2 in _CGO_LDFLAGS
					// but they don't mean anything to the linker so filter
					// them out.
					if flag != "-g" && !strings.HasPrefix(flag, "-O") {
						cgoldflags = append(cgoldflags, flag)
					}
				}
			}
		}
		return nil
	}

	readAndRemoveCgoFlags := func(archive string) (string, error) {
		newa, err := ioutil.TempFile(b.WorkDir, filepath.Base(archive))
		if err != nil {
			return "", err
		}
		olda, err := os.Open(archive)
		if err != nil {
			return "", err
		}
		_, err = io.Copy(newa, olda)
		if err != nil {
			return "", err
		}
		err = olda.Close()
		if err != nil {
			return "", err
		}
		err = newa.Close()
		if err != nil {
			return "", err
		}

		newarchive := newa.Name()
		err = b.run(b.WorkDir, desc, nil, "ar", "x", newarchive, "_cgo_flags")
		if err != nil {
			return "", err
		}
		err = b.run(".", desc, nil, "ar", "d", newarchive, "_cgo_flags")
		if err != nil {
			return "", err
		}
		err = readCgoFlags(filepath.Join(b.WorkDir, "_cgo_flags"))
		if err != nil {
			return "", err
		}
		return newarchive, nil
	}

	actionsSeen := make(map[*Action]bool)
	// Make a pre-order depth-first traversal of the action graph, taking note of
	// whether a shared library action has been seen on the way to an action (the
	// construction of the graph means that if any path to a node passes through
	// a shared library action, they all do).
	var walk func(a *Action, seenShlib bool)
	var err error
	walk = func(a *Action, seenShlib bool) {
		if actionsSeen[a] {
			return
		}
		actionsSeen[a] = true
		if a.Package != nil && !seenShlib {
			if a.Package.Standard {
				return
			}
			// We record the target of the first time we see a .a file
			// for a package to make sure that we prefer the 'install'
			// rather than the 'build' location (which may not exist any
			// more). We still need to traverse the dependencies of the
			// build action though so saying
			// if apackagePathsSeen[a.Package.ImportPath] { return }
			// doesn't work.
			if !apackagePathsSeen[a.Package.ImportPath] {
				apackagePathsSeen[a.Package.ImportPath] = true
				target := a.Target
				if len(a.Package.CgoFiles) > 0 || a.Package.UsesSwig() {
					target, err = readAndRemoveCgoFlags(target)
					if err != nil {
						return
					}
				}
				afiles = append(afiles, target)
			}
		}
		if strings.HasSuffix(a.Target, ".so") {
			shlibs = append(shlibs, a.Target)
			seenShlib = true
		}
		for _, a1 := range a.Deps {
			walk(a1, seenShlib)
			if err != nil {
				return
			}
		}
	}
	for _, a1 := range root.Deps {
		walk(a1, false)
		if err != nil {
			return err
		}
	}

	for _, a := range allactions {
		// Gather CgoLDFLAGS, but not from standard packages.
		// The go tool can dig up runtime/cgo from GOROOT and
		// think that it should use its CgoLDFLAGS, but gccgo
		// doesn't use runtime/cgo.
		if a.Package == nil {
			continue
		}
		if !a.Package.Standard {
			cgoldflags = append(cgoldflags, a.Package.CgoLDFLAGS...)
		}
		if len(a.Package.CgoFiles) > 0 {
			usesCgo = true
		}
		if a.Package.UsesSwig() {
			usesCgo = true
		}
		if len(a.Package.CXXFiles) > 0 || len(a.Package.SwigCXXFiles) > 0 {
			cxx = true
		}
		if len(a.Package.MFiles) > 0 {
			objc = true
		}
		if len(a.Package.FFiles) > 0 {
			fortran = true
		}
	}

	for i, o := range ofiles {
		if filepath.Base(o) == "_cgo_flags" {
			readCgoFlags(o)
			ofiles = append(ofiles[:i], ofiles[i+1:]...)
			break
		}
	}

	ldflags = append(ldflags, "-Wl,--whole-archive")
	ldflags = append(ldflags, afiles...)
	ldflags = append(ldflags, "-Wl,--no-whole-archive")

	ldflags = append(ldflags, cgoldflags...)
	ldflags = append(ldflags, envList("CGO_LDFLAGS", "")...)
	if root.Package != nil {
		ldflags = append(ldflags, root.Package.CgoLDFLAGS...)
	}

	ldflags = str.StringList("-Wl,-(", ldflags, "-Wl,-)")

	for _, shlib := range shlibs {
		ldflags = append(
			ldflags,
			"-L"+filepath.Dir(shlib),
			"-Wl,-rpath="+filepath.Dir(shlib),
			"-l"+strings.TrimSuffix(
				strings.TrimPrefix(filepath.Base(shlib), "lib"),
				".so"))
	}

	var realOut string
	switch buildmode {
	case "exe":
		if usesCgo && cfg.Goos == "linux" {
			ldflags = append(ldflags, "-Wl,-E")
		}

	case "c-archive":
		// Link the Go files into a single .o, and also link
		// in -lgolibbegin.
		//
		// We need to use --whole-archive with -lgolibbegin
		// because it doesn't define any symbols that will
		// cause the contents to be pulled in; it's just
		// initialization code.
		//
		// The user remains responsible for linking against
		// -lgo -lpthread -lm in the final link. We can't use
		// -r to pick them up because we can't combine
		// split-stack and non-split-stack code in a single -r
		// link, and libgo picks up non-split-stack code from
		// libffi.
		ldflags = append(ldflags, "-Wl,-r", "-nostdlib", "-Wl,--whole-archive", "-lgolibbegin", "-Wl,--no-whole-archive")

		if nopie := b.gccNoPie(); nopie != "" {
			ldflags = append(ldflags, nopie)
		}

		// We are creating an object file, so we don't want a build ID.
		ldflags = b.disableBuildID(ldflags)

		realOut = out
		out = out + ".o"

	case "c-shared":
		ldflags = append(ldflags, "-shared", "-nostdlib", "-Wl,--whole-archive", "-lgolibbegin", "-Wl,--no-whole-archive", "-lgo", "-lgcc_s", "-lgcc", "-lc", "-lgcc")
	case "shared":
		ldflags = append(ldflags, "-zdefs", "-shared", "-nostdlib", "-lgo", "-lgcc_s", "-lgcc", "-lc")

	default:
		base.Fatalf("-buildmode=%s not supported for gccgo", buildmode)
	}

	switch buildmode {
	case "exe", "c-shared":
		if cxx {
			ldflags = append(ldflags, "-lstdc++")
		}
		if objc {
			ldflags = append(ldflags, "-lobjc")
		}
		if fortran {
			fc := os.Getenv("FC")
			if fc == "" {
				fc = "gfortran"
			}
			// support gfortran out of the box and let others pass the correct link options
			// via CGO_LDFLAGS
			if strings.Contains(fc, "gfortran") {
				ldflags = append(ldflags, "-lgfortran")
			}
		}
	}

	if err := b.run(".", desc, nil, tools.linker(), "-o", out, ofiles, ldflags, buildGccgoflags); err != nil {
		return err
	}

	switch buildmode {
	case "c-archive":
		if err := b.run(".", desc, nil, "ar", "rc", realOut, out); err != nil {
			return err
		}
	}
	return nil
}

func (tools gccgoToolchain) ld(b *Builder, root *Action, out string, allactions []*Action, mainpkg string, ofiles []string) error {
	return tools.link(b, root, out, allactions, mainpkg, ofiles, ldBuildmode, root.Package.ImportPath)
}

func (tools gccgoToolchain) ldShared(b *Builder, toplevelactions []*Action, out string, allactions []*Action) error {
	fakeRoot := &Action{}
	fakeRoot.Deps = toplevelactions
	return tools.link(b, fakeRoot, out, allactions, "", nil, "shared", out)
}

func (tools gccgoToolchain) cc(b *Builder, p *load.Package, objdir, ofile, cfile string) error {
	inc := filepath.Join(cfg.GOROOT, "pkg", "include")
	cfile = mkAbs(p.Dir, cfile)
	defs := []string{"-D", "GOOS_" + cfg.Goos, "-D", "GOARCH_" + cfg.Goarch}
	defs = append(defs, b.gccArchArgs()...)
	if pkgpath := gccgoCleanPkgpath(p); pkgpath != "" {
		defs = append(defs, `-D`, `GOPKGPATH="`+pkgpath+`"`)
	}
	switch cfg.Goarch {
	case "386", "amd64":
		defs = append(defs, "-fsplit-stack")
	}
	defs = tools.maybePIC(defs)
	return b.run(p.Dir, p.ImportPath, nil, envList("CC", cfg.DefaultCC), "-Wall", "-g",
		"-I", objdir, "-I", inc, "-o", ofile, defs, "-c", cfile)
}

// maybePIC adds -fPIC to the list of arguments if needed.
func (tools gccgoToolchain) maybePIC(args []string) []string {
	switch cfg.BuildBuildmode {
	case "c-shared", "shared", "plugin":
		args = append(args, "-fPIC")
	}
	return args
}

func gccgoPkgpath(p *load.Package) string {
	if p.Internal.Build.IsCommand() && !p.Internal.ForceLibrary {
		return ""
	}
	return p.ImportPath
}

func gccgoCleanPkgpath(p *load.Package) string {
	clean := func(r rune) rune {
		switch {
		case 'A' <= r && r <= 'Z', 'a' <= r && r <= 'z',
			'0' <= r && r <= '9':
			return r
		}
		return '_'
	}
	return strings.Map(clean, gccgoPkgpath(p))
}

// gcc runs the gcc C compiler to create an object from a single C file.
func (b *Builder) gcc(p *load.Package, out string, flags []string, cfile string) error {
	return b.ccompile(p, out, flags, cfile, b.GccCmd(p.Dir))
}

// gxx runs the g++ C++ compiler to create an object from a single C++ file.
func (b *Builder) gxx(p *load.Package, out string, flags []string, cxxfile string) error {
	return b.ccompile(p, out, flags, cxxfile, b.GxxCmd(p.Dir))
}

// gfortran runs the gfortran Fortran compiler to create an object from a single Fortran file.
func (b *Builder) gfortran(p *load.Package, out string, flags []string, ffile string) error {
	return b.ccompile(p, out, flags, ffile, b.gfortranCmd(p.Dir))
}

// ccompile runs the given C or C++ compiler and creates an object from a single source file.
func (b *Builder) ccompile(p *load.Package, outfile string, flags []string, file string, compiler []string) error {
	file = mkAbs(p.Dir, file)
	desc := p.ImportPath
	if !filepath.IsAbs(outfile) {
		outfile = filepath.Join(p.Dir, outfile)
	}
	output, err := b.runOut(filepath.Dir(file), desc, nil, compiler, flags, "-o", outfile, "-c", filepath.Base(file))
	if len(output) > 0 {
		// On FreeBSD 11, when we pass -g to clang 3.8 it
		// invokes its internal assembler with -dwarf-version=2.
		// When it sees .section .note.GNU-stack, it warns
		// "DWARF2 only supports one section per compilation unit".
		// This warning makes no sense, since the section is empty,
		// but it confuses people.
		// We work around the problem by detecting the warning
		// and dropping -g and trying again.
		if bytes.Contains(output, []byte("DWARF2 only supports one section per compilation unit")) {
			newFlags := make([]string, 0, len(flags))
			for _, f := range flags {
				if !strings.HasPrefix(f, "-g") {
					newFlags = append(newFlags, f)
				}
			}
			if len(newFlags) < len(flags) {
				return b.ccompile(p, outfile, newFlags, file, compiler)
			}
		}

		b.showOutput(p.Dir, desc, b.processOutput(output))
		if err != nil {
			err = errPrintedOutput
		} else if os.Getenv("GO_BUILDER_NAME") != "" {
			return errors.New("C compiler warning promoted to error on Go builders")
		}
	}
	return err
}

// gccld runs the gcc linker to create an executable from a set of object files.
func (b *Builder) gccld(p *load.Package, out string, flags []string, obj []string) error {
	var cmd []string
	if len(p.CXXFiles) > 0 || len(p.SwigCXXFiles) > 0 {
		cmd = b.GxxCmd(p.Dir)
	} else {
		cmd = b.GccCmd(p.Dir)
	}
	return b.run(p.Dir, p.ImportPath, nil, cmd, "-o", out, obj, flags)
}

// gccCmd returns a gcc command line prefix
// defaultCC is defined in zdefaultcc.go, written by cmd/dist.
func (b *Builder) GccCmd(objdir string) []string {
	return b.ccompilerCmd("CC", cfg.DefaultCC, objdir)
}

// gxxCmd returns a g++ command line prefix
// defaultCXX is defined in zdefaultcc.go, written by cmd/dist.
func (b *Builder) GxxCmd(objdir string) []string {
	return b.ccompilerCmd("CXX", cfg.DefaultCXX, objdir)
}

// gfortranCmd returns a gfortran command line prefix.
func (b *Builder) gfortranCmd(objdir string) []string {
	return b.ccompilerCmd("FC", "gfortran", objdir)
}

// ccompilerCmd returns a command line prefix for the given environment
// variable and using the default command when the variable is empty.
func (b *Builder) ccompilerCmd(envvar, defcmd, objdir string) []string {
	// NOTE: env.go's mkEnv knows that the first three
	// strings returned are "gcc", "-I", objdir (and cuts them off).

	compiler := envList(envvar, defcmd)
	a := []string{compiler[0], "-I", objdir}
	a = append(a, compiler[1:]...)

	// Definitely want -fPIC but on Windows gcc complains
	// "-fPIC ignored for target (all code is position independent)"
	if cfg.Goos != "windows" {
		a = append(a, "-fPIC")
	}
	a = append(a, b.gccArchArgs()...)
	// gcc-4.5 and beyond require explicit "-pthread" flag
	// for multithreading with pthread library.
	if cfg.BuildContext.CgoEnabled {
		switch cfg.Goos {
		case "windows":
			a = append(a, "-mthreads")
		default:
			a = append(a, "-pthread")
		}
	}

	if strings.Contains(a[0], "clang") {
		// disable ASCII art in clang errors, if possible
		a = append(a, "-fno-caret-diagnostics")
		// clang is too smart about command-line arguments
		a = append(a, "-Qunused-arguments")
	}

	// disable word wrapping in error messages
	a = append(a, "-fmessage-length=0")

	// Tell gcc not to include the work directory in object files.
	if b.gccSupportsFlag("-fdebug-prefix-map=a=b") {
		a = append(a, "-fdebug-prefix-map="+b.WorkDir+"=/tmp/go-build")
	}

	// Tell gcc not to include flags in object files, which defeats the
	// point of -fdebug-prefix-map above.
	if b.gccSupportsFlag("-gno-record-gcc-switches") {
		a = append(a, "-gno-record-gcc-switches")
	}

	// On OS X, some of the compilers behave as if -fno-common
	// is always set, and the Mach-O linker in 6l/8l assumes this.
	// See https://golang.org/issue/3253.
	if cfg.Goos == "darwin" {
		a = append(a, "-fno-common")
	}

	return a
}

// gccNoPie returns the flag to use to request non-PIE. On systems
// with PIE (position independent executables) enabled by default,
// -no-pie must be passed when doing a partial link with -Wl,-r.
// But -no-pie is not supported by all compilers, and clang spells it -nopie.
func (b *Builder) gccNoPie() string {
	if b.gccSupportsFlag("-no-pie") {
		return "-no-pie"
	}
	if b.gccSupportsFlag("-nopie") {
		return "-nopie"
	}
	return ""
}

// gccSupportsFlag checks to see if the compiler supports a flag.
func (b *Builder) gccSupportsFlag(flag string) bool {
	b.exec.Lock()
	defer b.exec.Unlock()
	if b, ok := b.flagCache[flag]; ok {
		return b
	}
	if b.flagCache == nil {
		src := filepath.Join(b.WorkDir, "trivial.c")
		if err := ioutil.WriteFile(src, []byte{}, 0666); err != nil {
			return false
		}
		b.flagCache = make(map[string]bool)
	}
	cmdArgs := append(envList("CC", cfg.DefaultCC), flag, "-c", "trivial.c")
	if cfg.BuildN || cfg.BuildX {
		b.Showcmd(b.WorkDir, "%s", joinUnambiguously(cmdArgs))
		if cfg.BuildN {
			return false
		}
	}
	cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
	cmd.Dir = b.WorkDir
	cmd.Env = base.MergeEnvLists([]string{"LC_ALL=C"}, base.EnvForDir(cmd.Dir, os.Environ()))
	out, err := cmd.CombinedOutput()
	supported := err == nil && !bytes.Contains(out, []byte("unrecognized"))
	b.flagCache[flag] = supported
	return supported
}

// gccArchArgs returns arguments to pass to gcc based on the architecture.
func (b *Builder) gccArchArgs() []string {
	switch cfg.Goarch {
	case "386":
		return []string{"-m32"}
	case "amd64", "amd64p32":
		return []string{"-m64"}
	case "arm":
		return []string{"-marm"} // not thumb
	case "s390x":
		return []string{"-m64", "-march=z196"}
	case "mips64", "mips64le":
		return []string{"-mabi=64"}
	case "mips", "mipsle":
		return []string{"-mabi=32", "-march=mips32"}
	}
	return nil
}

// envList returns the value of the given environment variable broken
// into fields, using the default value when the variable is empty.
func envList(key, def string) []string {
	v := os.Getenv(key)
	if v == "" {
		v = def
	}
	return strings.Fields(v)
}

// CFlags returns the flags to use when invoking the C, C++ or Fortran compilers, or cgo.
func (b *Builder) CFlags(p *load.Package) (cppflags, cflags, cxxflags, fflags, ldflags []string) {
	defaults := "-g -O2"

	cppflags = str.StringList(envList("CGO_CPPFLAGS", ""), p.CgoCPPFLAGS)
	cflags = str.StringList(envList("CGO_CFLAGS", defaults), p.CgoCFLAGS)
	cxxflags = str.StringList(envList("CGO_CXXFLAGS", defaults), p.CgoCXXFLAGS)
	fflags = str.StringList(envList("CGO_FFLAGS", defaults), p.CgoFFLAGS)
	ldflags = str.StringList(envList("CGO_LDFLAGS", defaults), p.CgoLDFLAGS)
	return
}

var cgoRe = regexp.MustCompile(`[/\\:]`)

func (b *Builder) cgo(a *Action, cgoExe, obj string, pcCFLAGS, pcLDFLAGS, cgofiles, objdirCgofiles, gccfiles, gxxfiles, mfiles, ffiles []string) (outGo, outObj []string, err error) {
	p := a.Package
	cgoCPPFLAGS, cgoCFLAGS, cgoCXXFLAGS, cgoFFLAGS, cgoLDFLAGS := b.CFlags(p)
	cgoCPPFLAGS = append(cgoCPPFLAGS, pcCFLAGS...)
	cgoLDFLAGS = append(cgoLDFLAGS, pcLDFLAGS...)
	// If we are compiling Objective-C code, then we need to link against libobjc
	if len(mfiles) > 0 {
		cgoLDFLAGS = append(cgoLDFLAGS, "-lobjc")
	}

	// Likewise for Fortran, except there are many Fortran compilers.
	// Support gfortran out of the box and let others pass the correct link options
	// via CGO_LDFLAGS
	if len(ffiles) > 0 {
		fc := os.Getenv("FC")
		if fc == "" {
			fc = "gfortran"
		}
		if strings.Contains(fc, "gfortran") {
			cgoLDFLAGS = append(cgoLDFLAGS, "-lgfortran")
		}
	}

	if cfg.BuildMSan {
		cgoCFLAGS = append([]string{"-fsanitize=memory"}, cgoCFLAGS...)
		cgoLDFLAGS = append([]string{"-fsanitize=memory"}, cgoLDFLAGS...)
	}

	// Allows including _cgo_export.h from .[ch] files in the package.
	cgoCPPFLAGS = append(cgoCPPFLAGS, "-I", obj)

	// If we have cgo files in the object directory, then copy any
	// other cgo files into the object directory, and pass a
	// -srcdir option to cgo.
	var srcdirarg []string
	if len(objdirCgofiles) > 0 {
		for _, fn := range cgofiles {
			if err := b.copyFile(a, obj+filepath.Base(fn), filepath.Join(p.Dir, fn), 0666, false); err != nil {
				return nil, nil, err
			}
		}
		cgofiles = append(cgofiles, objdirCgofiles...)
		srcdirarg = []string{"-srcdir", obj}
	}

	// cgo
	// TODO: CGO_FLAGS?
	gofiles := []string{obj + "_cgo_gotypes.go"}
	cfiles := []string{"_cgo_export.c"}
	for _, fn := range cgofiles {
		f := cgoRe.ReplaceAllString(fn[:len(fn)-2], "_")
		gofiles = append(gofiles, obj+f+"cgo1.go")
		cfiles = append(cfiles, f+"cgo2.c")
	}

	// TODO: make cgo not depend on $GOARCH?

	cgoflags := []string{}
	if p.Standard && p.ImportPath == "runtime/cgo" {
		cgoflags = append(cgoflags, "-import_runtime_cgo=false")
	}
	if p.Standard && (p.ImportPath == "runtime/race" || p.ImportPath == "runtime/msan" || p.ImportPath == "runtime/cgo") {
		cgoflags = append(cgoflags, "-import_syscall=false")
	}

	// Update $CGO_LDFLAGS with p.CgoLDFLAGS.
	var cgoenv []string
	if len(cgoLDFLAGS) > 0 {
		flags := make([]string, len(cgoLDFLAGS))
		for i, f := range cgoLDFLAGS {
			flags[i] = strconv.Quote(f)
		}
		cgoenv = []string{"CGO_LDFLAGS=" + strings.Join(flags, " ")}
	}

	if cfg.BuildToolchainName == "gccgo" {
		switch cfg.Goarch {
		case "386", "amd64":
			cgoCFLAGS = append(cgoCFLAGS, "-fsplit-stack")
		}
		cgoflags = append(cgoflags, "-gccgo")
		if pkgpath := gccgoPkgpath(p); pkgpath != "" {
			cgoflags = append(cgoflags, "-gccgopkgpath="+pkgpath)
		}
	}

	switch cfg.BuildBuildmode {
	case "c-archive", "c-shared":
		// Tell cgo that if there are any exported functions
		// it should generate a header file that C code can
		// #include.
		cgoflags = append(cgoflags, "-exportheader="+obj+"_cgo_install.h")
	}

	if err := b.run(p.Dir, p.ImportPath, cgoenv, cfg.BuildToolexec, cgoExe, srcdirarg, "-objdir", obj, "-importpath", p.ImportPath, cgoflags, "--", cgoCPPFLAGS, cgoCFLAGS, cgofiles); err != nil {
		return nil, nil, err
	}
	outGo = append(outGo, gofiles...)

	// gcc
	cflags := str.StringList(cgoCPPFLAGS, cgoCFLAGS)
	for _, cfile := range cfiles {
		ofile := obj + cfile[:len(cfile)-1] + "o"
		if err := b.gcc(p, ofile, cflags, obj+cfile); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, ofile)
	}

	for _, file := range gccfiles {
		base := filepath.Base(file)
		ofile := obj + cgoRe.ReplaceAllString(base[:len(base)-1], "_") + "o"
		if err := b.gcc(p, ofile, cflags, file); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, ofile)
	}

	cxxflags := str.StringList(cgoCPPFLAGS, cgoCXXFLAGS)
	for _, file := range gxxfiles {
		// Append .o to the file, just in case the pkg has file.c and file.cpp
		ofile := obj + cgoRe.ReplaceAllString(filepath.Base(file), "_") + ".o"
		if err := b.gxx(p, ofile, cxxflags, file); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, ofile)
	}

	for _, file := range mfiles {
		// Append .o to the file, just in case the pkg has file.c and file.m
		ofile := obj + cgoRe.ReplaceAllString(filepath.Base(file), "_") + ".o"
		if err := b.gcc(p, ofile, cflags, file); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, ofile)
	}

	fflags := str.StringList(cgoCPPFLAGS, cgoFFLAGS)
	for _, file := range ffiles {
		// Append .o to the file, just in case the pkg has file.c and file.f
		ofile := obj + cgoRe.ReplaceAllString(filepath.Base(file), "_") + ".o"
		if err := b.gfortran(p, ofile, fflags, file); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, ofile)
	}

	switch cfg.BuildToolchainName {
	case "gc":
		importGo := obj + "_cgo_import.go"
		if err := b.dynimport(p, obj, importGo, cgoExe, cflags, cgoLDFLAGS, outObj); err != nil {
			return nil, nil, err
		}
		outGo = append(outGo, importGo)

		ofile := obj + "_all.o"
		if err := b.collect(p, obj, ofile, cgoLDFLAGS, outObj); err != nil {
			return nil, nil, err
		}
		outObj = []string{ofile}

	case "gccgo":
		defunC := obj + "_cgo_defun.c"
		defunObj := obj + "_cgo_defun.o"
		if err := BuildToolchain.cc(b, p, obj, defunObj, defunC); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, defunObj)

	default:
		noCompiler()
	}

	return outGo, outObj, nil
}

// dynimport creates a Go source file named importGo containing
// //go:cgo_import_dynamic directives for each symbol or library
// dynamically imported by the object files outObj.
func (b *Builder) dynimport(p *load.Package, obj, importGo, cgoExe string, cflags, cgoLDFLAGS, outObj []string) error {
	cfile := obj + "_cgo_main.c"
	ofile := obj + "_cgo_main.o"
	if err := b.gcc(p, ofile, cflags, cfile); err != nil {
		return err
	}

	linkobj := str.StringList(ofile, outObj, p.SysoFiles)
	dynobj := obj + "_cgo_.o"

	// we need to use -pie for Linux/ARM to get accurate imported sym
	ldflags := cgoLDFLAGS
	if (cfg.Goarch == "arm" && cfg.Goos == "linux") || cfg.Goos == "android" {
		ldflags = append(ldflags, "-pie")
	}
	if err := b.gccld(p, dynobj, ldflags, linkobj); err != nil {
		return err
	}

	// cgo -dynimport
	var cgoflags []string
	if p.Standard && p.ImportPath == "runtime/cgo" {
		cgoflags = []string{"-dynlinker"} // record path to dynamic linker
	}
	return b.run(p.Dir, p.ImportPath, nil, cfg.BuildToolexec, cgoExe, "-dynpackage", p.Name, "-dynimport", dynobj, "-dynout", importGo, cgoflags)
}

// collect partially links the object files outObj into a single
// relocatable object file named ofile.
func (b *Builder) collect(p *load.Package, obj, ofile string, cgoLDFLAGS, outObj []string) error {
	// When linking relocatable objects, various flags need to be
	// filtered out as they are inapplicable and can cause some linkers
	// to fail.
	var ldflags []string
	for i := 0; i < len(cgoLDFLAGS); i++ {
		f := cgoLDFLAGS[i]
		switch {
		// skip "-lc" or "-l somelib"
		case strings.HasPrefix(f, "-l"):
			if f == "-l" {
				i++
			}
		// skip "-framework X" on Darwin
		case cfg.Goos == "darwin" && f == "-framework":
			i++
		// skip "*.{dylib,so,dll,o,a}"
		case strings.HasSuffix(f, ".dylib"),
			strings.HasSuffix(f, ".so"),
			strings.HasSuffix(f, ".dll"),
			strings.HasSuffix(f, ".o"),
			strings.HasSuffix(f, ".a"):
		// Remove any -fsanitize=foo flags.
		// Otherwise the compiler driver thinks that we are doing final link
		// and links sanitizer runtime into the object file. But we are not doing
		// the final link, we will link the resulting object file again. And
		// so the program ends up with two copies of sanitizer runtime.
		// See issue 8788 for details.
		case strings.HasPrefix(f, "-fsanitize="):
			continue
		// runpath flags not applicable unless building a shared
		// object or executable; see issue 12115 for details. This
		// is necessary as Go currently does not offer a way to
		// specify the set of LDFLAGS that only apply to shared
		// objects.
		case strings.HasPrefix(f, "-Wl,-rpath"):
			if f == "-Wl,-rpath" || f == "-Wl,-rpath-link" {
				// Skip following argument to -rpath* too.
				i++
			}
		default:
			ldflags = append(ldflags, f)
		}
	}

	ldflags = append(ldflags, "-Wl,-r", "-nostdlib")

	if flag := b.gccNoPie(); flag != "" {
		ldflags = append(ldflags, flag)
	}

	// We are creating an object file, so we don't want a build ID.
	ldflags = b.disableBuildID(ldflags)

	return b.gccld(p, ofile, ldflags, outObj)
}

// Run SWIG on all SWIG input files.
// TODO: Don't build a shared library, once SWIG emits the necessary
// pragmas for external linking.
func (b *Builder) swig(p *load.Package, obj string, pcCFLAGS []string) (outGo, outC, outCXX []string, err error) {
	if err := b.swigVersionCheck(); err != nil {
		return nil, nil, nil, err
	}

	intgosize, err := b.swigIntSize(obj)
	if err != nil {
		return nil, nil, nil, err
	}

	for _, f := range p.SwigFiles {
		goFile, cFile, err := b.swigOne(p, f, obj, pcCFLAGS, false, intgosize)
		if err != nil {
			return nil, nil, nil, err
		}
		if goFile != "" {
			outGo = append(outGo, goFile)
		}
		if cFile != "" {
			outC = append(outC, cFile)
		}
	}
	for _, f := range p.SwigCXXFiles {
		goFile, cxxFile, err := b.swigOne(p, f, obj, pcCFLAGS, true, intgosize)
		if err != nil {
			return nil, nil, nil, err
		}
		if goFile != "" {
			outGo = append(outGo, goFile)
		}
		if cxxFile != "" {
			outCXX = append(outCXX, cxxFile)
		}
	}
	return outGo, outC, outCXX, nil
}

// Make sure SWIG is new enough.
var (
	swigCheckOnce sync.Once
	swigCheck     error
)

func (b *Builder) swigDoVersionCheck() error {
	out, err := b.runOut("", "", nil, "swig", "-version")
	if err != nil {
		return err
	}
	re := regexp.MustCompile(`[vV]ersion +([\d]+)([.][\d]+)?([.][\d]+)?`)
	matches := re.FindSubmatch(out)
	if matches == nil {
		// Can't find version number; hope for the best.
		return nil
	}

	major, err := strconv.Atoi(string(matches[1]))
	if err != nil {
		// Can't find version number; hope for the best.
		return nil
	}
	const errmsg = "must have SWIG version >= 3.0.6"
	if major < 3 {
		return errors.New(errmsg)
	}
	if major > 3 {
		// 4.0 or later
		return nil
	}

	// We have SWIG version 3.x.
	if len(matches[2]) > 0 {
		minor, err := strconv.Atoi(string(matches[2][1:]))
		if err != nil {
			return nil
		}
		if minor > 0 {
			// 3.1 or later
			return nil
		}
	}

	// We have SWIG version 3.0.x.
	if len(matches[3]) > 0 {
		patch, err := strconv.Atoi(string(matches[3][1:]))
		if err != nil {
			return nil
		}
		if patch < 6 {
			// Before 3.0.6.
			return errors.New(errmsg)
		}
	}

	return nil
}

func (b *Builder) swigVersionCheck() error {
	swigCheckOnce.Do(func() {
		swigCheck = b.swigDoVersionCheck()
	})
	return swigCheck
}

// Find the value to pass for the -intgosize option to swig.
var (
	swigIntSizeOnce  sync.Once
	swigIntSize      string
	swigIntSizeError error
)

// This code fails to build if sizeof(int) <= 32
const swigIntSizeCode = `
package main
const i int = 1 << 32
`

// Determine the size of int on the target system for the -intgosize option
// of swig >= 2.0.9. Run only once.
func (b *Builder) swigDoIntSize(obj string) (intsize string, err error) {
	if cfg.BuildN {
		return "$INTBITS", nil
	}
	src := filepath.Join(b.WorkDir, "swig_intsize.go")
	if err = ioutil.WriteFile(src, []byte(swigIntSizeCode), 0666); err != nil {
		return
	}
	srcs := []string{src}

	p := load.GoFilesPackage(srcs)

	if _, _, e := BuildToolchain.gc(b, p, "", obj, false, nil, srcs); e != nil {
		return "32", nil
	}
	return "64", nil
}

// Determine the size of int on the target system for the -intgosize option
// of swig >= 2.0.9.
func (b *Builder) swigIntSize(obj string) (intsize string, err error) {
	swigIntSizeOnce.Do(func() {
		swigIntSize, swigIntSizeError = b.swigDoIntSize(obj)
	})
	return swigIntSize, swigIntSizeError
}

// Run SWIG on one SWIG input file.
func (b *Builder) swigOne(p *load.Package, file, obj string, pcCFLAGS []string, cxx bool, intgosize string) (outGo, outC string, err error) {
	cgoCPPFLAGS, cgoCFLAGS, cgoCXXFLAGS, _, _ := b.CFlags(p)
	var cflags []string
	if cxx {
		cflags = str.StringList(cgoCPPFLAGS, pcCFLAGS, cgoCXXFLAGS)
	} else {
		cflags = str.StringList(cgoCPPFLAGS, pcCFLAGS, cgoCFLAGS)
	}

	n := 5 // length of ".swig"
	if cxx {
		n = 8 // length of ".swigcxx"
	}
	base := file[:len(file)-n]
	goFile := base + ".go"
	gccBase := base + "_wrap."
	gccExt := "c"
	if cxx {
		gccExt = "cxx"
	}

	gccgo := cfg.BuildToolchainName == "gccgo"

	// swig
	args := []string{
		"-go",
		"-cgo",
		"-intgosize", intgosize,
		"-module", base,
		"-o", obj + gccBase + gccExt,
		"-outdir", obj,
	}

	for _, f := range cflags {
		if len(f) > 3 && f[:2] == "-I" {
			args = append(args, f)
		}
	}

	if gccgo {
		args = append(args, "-gccgo")
		if pkgpath := gccgoPkgpath(p); pkgpath != "" {
			args = append(args, "-go-pkgpath", pkgpath)
		}
	}
	if cxx {
		args = append(args, "-c++")
	}

	out, err := b.runOut(p.Dir, p.ImportPath, nil, "swig", args, file)
	if err != nil {
		if len(out) > 0 {
			if bytes.Contains(out, []byte("-intgosize")) || bytes.Contains(out, []byte("-cgo")) {
				return "", "", errors.New("must have SWIG version >= 3.0.6")
			}
			b.showOutput(p.Dir, p.ImportPath, b.processOutput(out)) // swig error
			return "", "", errPrintedOutput
		}
		return "", "", err
	}
	if len(out) > 0 {
		b.showOutput(p.Dir, p.ImportPath, b.processOutput(out)) // swig warning
	}

	return goFile, obj + gccBase + gccExt, nil
}

// disableBuildID adjusts a linker command line to avoid creating a
// build ID when creating an object file rather than an executable or
// shared library. Some systems, such as Ubuntu, always add
// --build-id to every link, but we don't want a build ID when we are
// producing an object file. On some of those system a plain -r (not
// -Wl,-r) will turn off --build-id, but clang 3.0 doesn't support a
// plain -r. I don't know how to turn off --build-id when using clang
// other than passing a trailing --build-id=none. So that is what we
// do, but only on systems likely to support it, which is to say,
// systems that normally use gold or the GNU linker.
func (b *Builder) disableBuildID(ldflags []string) []string {
	switch cfg.Goos {
	case "android", "dragonfly", "linux", "netbsd":
		ldflags = append(ldflags, "-Wl,--build-id=none")
	}
	return ldflags
}

// An actionQueue is a priority queue of actions.
type actionQueue []*Action

// Implement heap.Interface
func (q *actionQueue) Len() int           { return len(*q) }
func (q *actionQueue) Swap(i, j int)      { (*q)[i], (*q)[j] = (*q)[j], (*q)[i] }
func (q *actionQueue) Less(i, j int) bool { return (*q)[i].priority < (*q)[j].priority }
func (q *actionQueue) Push(x interface{}) { *q = append(*q, x.(*Action)) }
func (q *actionQueue) Pop() interface{} {
	n := len(*q) - 1
	x := (*q)[n]
	*q = (*q)[:n]
	return x
}

func (q *actionQueue) push(a *Action) {
	heap.Push(q, a)
}

func (q *actionQueue) pop() *Action {
	return heap.Pop(q).(*Action)
}

func InstrumentInit() {
	if !cfg.BuildRace && !cfg.BuildMSan {
		return
	}
	if cfg.BuildRace && cfg.BuildMSan {
		fmt.Fprintf(os.Stderr, "go %s: may not use -race and -msan simultaneously\n", flag.Args()[0])
		os.Exit(2)
	}
	if cfg.BuildMSan && (cfg.Goos != "linux" || cfg.Goarch != "amd64") {
		fmt.Fprintf(os.Stderr, "-msan is not supported on %s/%s\n", cfg.Goos, cfg.Goarch)
		os.Exit(2)
	}
	if cfg.Goarch != "amd64" || cfg.Goos != "linux" && cfg.Goos != "freebsd" && cfg.Goos != "darwin" && cfg.Goos != "windows" {
		fmt.Fprintf(os.Stderr, "go %s: -race and -msan are only supported on linux/amd64, freebsd/amd64, darwin/amd64 and windows/amd64\n", flag.Args()[0])
		os.Exit(2)
	}
	if !cfg.BuildContext.CgoEnabled {
		fmt.Fprintf(os.Stderr, "go %s: -race requires cgo; enable cgo by setting CGO_ENABLED=1\n", flag.Args()[0])
		os.Exit(2)
	}
	if cfg.BuildRace {
		buildGcflags = append(buildGcflags, "-race")
		cfg.BuildLdflags = append(cfg.BuildLdflags, "-race")
	} else {
		buildGcflags = append(buildGcflags, "-msan")
		cfg.BuildLdflags = append(cfg.BuildLdflags, "-msan")
	}
	if cfg.BuildContext.InstallSuffix != "" {
		cfg.BuildContext.InstallSuffix += "_"
	}

	if cfg.BuildRace {
		cfg.BuildContext.InstallSuffix += "race"
		cfg.BuildContext.BuildTags = append(cfg.BuildContext.BuildTags, "race")
	} else {
		cfg.BuildContext.InstallSuffix += "msan"
		cfg.BuildContext.BuildTags = append(cfg.BuildContext.BuildTags, "msan")
	}
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
