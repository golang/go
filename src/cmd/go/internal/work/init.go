// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Build initialization (after flag parsing).

package work

import (
	"bytes"
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/modload"
	"cmd/internal/quoted"
	"fmt"
	"internal/platform"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"sync"
)

var buildInitStarted = false

// makeCfgChangedEnv is the environment to set to
// override the current environment for GOOS, GOARCH, and the GOARCH-specific
// architecture environment variable to the configuration used by
// the go command. They may be different because go tool <tool> for builtin
// tools need to be built using the host configuration, so the configuration
// used will be changed from that set in the environment. It is clipped
// so its can append to it without changing it.
var cfgChangedEnv []string

func makeCfgChangedEnv() []string {
	var env []string
	if cfg.Getenv("GOOS") != cfg.Goos {
		env = append(env, "GOOS="+cfg.Goos)
	}
	if cfg.Getenv("GOARCH") != cfg.Goarch {
		env = append(env, "GOARCH="+cfg.Goarch)
	}
	if archenv, val, changed := cfg.GetArchEnv(); changed {
		env = append(env, archenv+"="+val)
	}
	return slices.Clip(env)
}

func BuildInit(loaderstate *modload.State) {
	if buildInitStarted {
		base.Fatalf("go: internal error: work.BuildInit called more than once")
	}
	buildInitStarted = true
	base.AtExit(closeBuilders)

	modload.Init(loaderstate)
	instrumentInit()
	buildModeInit()
	cfgChangedEnv = makeCfgChangedEnv()

	if err := fsys.Init(); err != nil {
		base.Fatal(err)
	}
	if from, replaced := fsys.DirContainsReplacement(cfg.GOMODCACHE); replaced {
		base.Fatalf("go: overlay contains a replacement for %s. Files beneath GOMODCACHE (%s) must not be replaced.", from, cfg.GOMODCACHE)
	}

	// Make sure -pkgdir is absolute, because we run commands
	// in different directories.
	if cfg.BuildPkgdir != "" && !filepath.IsAbs(cfg.BuildPkgdir) {
		p, err := filepath.Abs(cfg.BuildPkgdir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "go: evaluating -pkgdir: %v\n", err)
			base.SetExitStatus(2)
			base.Exit()
		}
		cfg.BuildPkgdir = p
	}

	if cfg.BuildP <= 0 {
		base.Fatalf("go: -p must be a positive integer: %v\n", cfg.BuildP)
	}

	// Make sure CC, CXX, and FC are absolute paths.
	for _, key := range []string{"CC", "CXX", "FC"} {
		value := cfg.Getenv(key)
		args, err := quoted.Split(value)
		if err != nil {
			base.Fatalf("go: %s environment variable could not be parsed: %v", key, err)
		}
		if len(args) == 0 {
			continue
		}
		path := args[0]
		if !filepath.IsAbs(path) && path != filepath.Base(path) {
			base.Fatalf("go: %s environment variable is relative; must be absolute path: %s\n", key, path)
		}
	}

	// Set covermode if not already set.
	// Ensure that -race and -covermode are compatible.
	if cfg.BuildCoverMode == "" {
		cfg.BuildCoverMode = "set"
		if cfg.BuildRace {
			// Default coverage mode is atomic when -race is set.
			cfg.BuildCoverMode = "atomic"
		}
	}
	if cfg.BuildRace && cfg.BuildCoverMode != "atomic" {
		base.Fatalf(`-covermode must be "atomic", not %q, when -race is enabled`, cfg.BuildCoverMode)
	}
}

// fuzzInstrumentFlags returns compiler flags that enable fuzzing instrumentation
// on supported platforms.
//
// On unsupported platforms, fuzzInstrumentFlags returns nil, meaning no
// instrumentation is added. 'go test -fuzz' still works without coverage,
// but it generates random inputs without guidance, so it's much less effective.
func fuzzInstrumentFlags() []string {
	if !platform.FuzzInstrumented(cfg.Goos, cfg.Goarch) {
		return nil
	}
	return []string{"-d=libfuzzer"}
}

func instrumentInit() {
	if !cfg.BuildRace && !cfg.BuildMSan && !cfg.BuildASan {
		return
	}
	if cfg.BuildRace && cfg.BuildMSan {
		fmt.Fprintf(os.Stderr, "go: may not use -race and -msan simultaneously\n")
		base.SetExitStatus(2)
		base.Exit()
	}
	if cfg.BuildRace && cfg.BuildASan {
		fmt.Fprintf(os.Stderr, "go: may not use -race and -asan simultaneously\n")
		base.SetExitStatus(2)
		base.Exit()
	}
	if cfg.BuildMSan && cfg.BuildASan {
		fmt.Fprintf(os.Stderr, "go: may not use -msan and -asan simultaneously\n")
		base.SetExitStatus(2)
		base.Exit()
	}
	if cfg.BuildMSan && !platform.MSanSupported(cfg.Goos, cfg.Goarch) {
		fmt.Fprintf(os.Stderr, "-msan is not supported on %s/%s\n", cfg.Goos, cfg.Goarch)
		base.SetExitStatus(2)
		base.Exit()
	}
	if cfg.BuildRace && !platform.RaceDetectorSupported(cfg.Goos, cfg.Goarch) {
		fmt.Fprintf(os.Stderr, "-race is not supported on %s/%s\n", cfg.Goos, cfg.Goarch)
		base.SetExitStatus(2)
		base.Exit()
	}
	if cfg.BuildASan && !platform.ASanSupported(cfg.Goos, cfg.Goarch) {
		fmt.Fprintf(os.Stderr, "-asan is not supported on %s/%s\n", cfg.Goos, cfg.Goarch)
		base.SetExitStatus(2)
		base.Exit()
	}
	// The current implementation is only compatible with the ASan library from version
	// v7 to v9 (See the description in src/runtime/asan/asan.go). Therefore, using the
	// -asan option must use a compatible version of ASan library, which requires that
	// the gcc version is not less than 7 and the clang version is not less than 9,
	// otherwise a segmentation fault will occur.
	if cfg.BuildASan {
		if err := compilerRequiredAsanVersion(); err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			base.SetExitStatus(2)
			base.Exit()
		}
	}

	mode := "race"
	if cfg.BuildMSan {
		mode = "msan"
		// MSAN needs PIE on all platforms except linux/amd64.
		// https://github.com/llvm/llvm-project/blob/llvmorg-13.0.1/clang/lib/Driver/SanitizerArgs.cpp#L621
		if cfg.BuildBuildmode == "default" && (cfg.Goos != "linux" || cfg.Goarch != "amd64") {
			cfg.BuildBuildmode = "pie"
		}
	}
	if cfg.BuildASan {
		mode = "asan"
	}
	modeFlag := "-" + mode

	// Check that cgo is enabled.
	// Note: On macOS, -race does not require cgo. -asan and -msan still do.
	if !cfg.BuildContext.CgoEnabled && (cfg.Goos != "darwin" || cfg.BuildASan || cfg.BuildMSan) {
		if runtime.GOOS != cfg.Goos || runtime.GOARCH != cfg.Goarch {
			fmt.Fprintf(os.Stderr, "go: %s requires cgo\n", modeFlag)
		} else {
			fmt.Fprintf(os.Stderr, "go: %s requires cgo; enable cgo by setting CGO_ENABLED=1\n", modeFlag)
		}

		base.SetExitStatus(2)
		base.Exit()
	}
	forcedGcflags = append(forcedGcflags, modeFlag)
	forcedLdflags = append(forcedLdflags, modeFlag)

	if cfg.BuildContext.InstallSuffix != "" {
		cfg.BuildContext.InstallSuffix += "_"
	}
	cfg.BuildContext.InstallSuffix += mode
	cfg.BuildContext.ToolTags = append(cfg.BuildContext.ToolTags, mode)
}

func buildModeInit() {
	gccgo := cfg.BuildToolchainName == "gccgo"
	var codegenArg string

	// Configure the build mode first, then verify that it is supported.
	// That way, if the flag is completely bogus we will prefer to error out with
	// "-buildmode=%s not supported" instead of naming the specific platform.

	switch cfg.BuildBuildmode {
	case "archive":
		pkgsFilter = pkgsNotMain
	case "c-archive":
		pkgsFilter = oneMainPkg
		if gccgo {
			codegenArg = "-fPIC"
		} else {
			switch cfg.Goos {
			case "darwin", "ios":
				switch cfg.Goarch {
				case "arm64":
					codegenArg = "-shared"
				}

			case "dragonfly", "freebsd", "illumos", "linux", "netbsd", "openbsd", "solaris":
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
			switch cfg.Goos {
			case "linux", "android", "freebsd":
				codegenArg = "-shared"
			case "windows":
				// Do not add usual .exe suffix to the .dll file.
				cfg.ExeSuffix = ""
			}
		}
		ldBuildmode = "c-shared"
	case "default":
		ldBuildmode = "exe"
		if platform.DefaultPIE(cfg.Goos, cfg.Goarch, cfg.BuildRace) {
			ldBuildmode = "pie"
			if cfg.Goos != "windows" && !gccgo {
				codegenArg = "-shared"
			}
		}
	case "exe":
		pkgsFilter = pkgsMain
		ldBuildmode = "exe"
		// Set the pkgsFilter to oneMainPkg if the user passed a specific binary output
		// and is using buildmode=exe for a better error message.
		// See issue #20017.
		if cfg.BuildO != "" {
			pkgsFilter = oneMainPkg
		}
	case "pie":
		if cfg.BuildRace && !platform.DefaultPIE(cfg.Goos, cfg.Goarch, cfg.BuildRace) {
			base.Fatalf("-buildmode=pie not supported when -race is enabled on %s/%s", cfg.Goos, cfg.Goarch)
		}
		if gccgo {
			codegenArg = "-fPIE"
		} else {
			switch cfg.Goos {
			case "aix", "windows":
			default:
				codegenArg = "-shared"
			}
		}
		ldBuildmode = "pie"
	case "shared":
		pkgsFilter = pkgsNotMain
		if gccgo {
			codegenArg = "-fPIC"
		} else {
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
			codegenArg = "-dynlink"
		}
		cfg.ExeSuffix = ".so"
		ldBuildmode = "plugin"
	default:
		base.Fatalf("buildmode=%s not supported", cfg.BuildBuildmode)
	}

	if cfg.BuildBuildmode != "default" && !platform.BuildModeSupported(cfg.BuildToolchainName, cfg.BuildBuildmode, cfg.Goos, cfg.Goarch) {
		base.Fatalf("-buildmode=%s not supported on %s/%s\n", cfg.BuildBuildmode, cfg.Goos, cfg.Goarch)
	}

	if cfg.BuildLinkshared {
		if !platform.BuildModeSupported(cfg.BuildToolchainName, "shared", cfg.Goos, cfg.Goarch) {
			base.Fatalf("-linkshared not supported on %s/%s\n", cfg.Goos, cfg.Goarch)
		}
		if gccgo {
			codegenArg = "-fPIC"
		} else {
			forcedAsmflags = append(forcedAsmflags, "-D=GOBUILDMODE_shared=1",
				"-linkshared")
			codegenArg = "-dynlink"
			forcedGcflags = append(forcedGcflags, "-linkshared")
			// TODO(mwhudson): remove -w when that gets fixed in linker.
			forcedLdflags = append(forcedLdflags, "-linkshared", "-w")
		}
	}
	if codegenArg != "" {
		if gccgo {
			forcedGccgoflags = append([]string{codegenArg}, forcedGccgoflags...)
		} else {
			forcedAsmflags = append([]string{codegenArg}, forcedAsmflags...)
			forcedGcflags = append([]string{codegenArg}, forcedGcflags...)
		}
		// Don't alter InstallSuffix when modifying default codegen args.
		if cfg.BuildBuildmode != "default" || cfg.BuildLinkshared {
			if cfg.BuildContext.InstallSuffix != "" {
				cfg.BuildContext.InstallSuffix += "_"
			}
			cfg.BuildContext.InstallSuffix += codegenArg[1:]
		}
	}

	switch cfg.BuildMod {
	case "":
		// Behavior will be determined automatically, as if no flag were passed.
	case "readonly", "vendor", "mod":
		if !cfg.ModulesEnabled && !base.InGOFLAGS("-mod") {
			base.Fatalf("build flag -mod=%s only valid when using modules", cfg.BuildMod)
		}
	default:
		base.Fatalf("-mod=%s not supported (can be '', 'mod', 'readonly', or 'vendor')", cfg.BuildMod)
	}
	if !cfg.ModulesEnabled {
		if cfg.ModCacheRW && !base.InGOFLAGS("-modcacherw") {
			base.Fatalf("build flag -modcacherw only valid when using modules")
		}
		if cfg.ModFile != "" && !base.InGOFLAGS("-mod") {
			base.Fatalf("build flag -modfile only valid when using modules")
		}
	}
}

type version struct {
	name         string
	major, minor int
}

var compiler struct {
	sync.Once
	version
	err error
}

// compilerVersion detects the version of $(go env CC).
// It returns a non-nil error if the compiler matches a known version schema but
// the version could not be parsed, or if $(go env CC) could not be determined.
func compilerVersion() (version, error) {
	compiler.Once.Do(func() {
		compiler.err = func() error {
			compiler.name = "unknown"
			cc := os.Getenv("CC")
			cmd := exec.Command(cc, "--version")
			cmd.Env = append(cmd.Environ(), "LANG=C")
			out, err := cmd.Output()
			if err != nil {
				// Compiler does not support "--version" flag: not Clang or GCC.
				return err
			}

			var match [][]byte
			if bytes.HasPrefix(out, []byte("gcc")) {
				compiler.name = "gcc"
				cmd := exec.Command(cc, "-v")
				cmd.Env = append(cmd.Environ(), "LANG=C")
				out, err := cmd.CombinedOutput()
				if err != nil {
					// gcc, but does not support gcc's "-v" flag?!
					return err
				}
				gccRE := regexp.MustCompile(`gcc version (\d+)\.(\d+)`)
				match = gccRE.FindSubmatch(out)
			} else {
				clangRE := regexp.MustCompile(`clang version (\d+)\.(\d+)`)
				if match = clangRE.FindSubmatch(out); len(match) > 0 {
					compiler.name = "clang"
				}
			}

			if len(match) < 3 {
				return nil // "unknown"
			}
			if compiler.major, err = strconv.Atoi(string(match[1])); err != nil {
				return err
			}
			if compiler.minor, err = strconv.Atoi(string(match[2])); err != nil {
				return err
			}
			return nil
		}()
	})
	return compiler.version, compiler.err
}

// compilerRequiredAsanVersion is a copy of the function defined in
// cmd/cgo/internal/testsanitizers/cc_test.go
// compilerRequiredAsanVersion reports whether the compiler is the version
// required by Asan.
func compilerRequiredAsanVersion() error {
	compiler, err := compilerVersion()
	if err != nil {
		return fmt.Errorf("-asan: the version of $(go env CC) could not be parsed")
	}

	switch compiler.name {
	case "gcc":
		if runtime.GOARCH == "ppc64le" && compiler.major < 9 {
			return fmt.Errorf("-asan is not supported with %s compiler %d.%d\n", compiler.name, compiler.major, compiler.minor)
		}
		if compiler.major < 7 {
			return fmt.Errorf("-asan is not supported with %s compiler %d.%d\n", compiler.name, compiler.major, compiler.minor)
		}
	case "clang":
		if compiler.major < 9 {
			return fmt.Errorf("-asan is not supported with %s compiler %d.%d\n", compiler.name, compiler.major, compiler.minor)
		}
	default:
		return fmt.Errorf("-asan: C compiler is not gcc or clang")
	}
	return nil
}
