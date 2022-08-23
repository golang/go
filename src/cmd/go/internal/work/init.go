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
	"cmd/internal/sys"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"sync"
)

var buildInitStarted = false

func BuildInit() {
	if buildInitStarted {
		base.Fatalf("go: internal error: work.BuildInit called more than once")
	}
	buildInitStarted = true
	base.AtExit(closeBuilders)

	modload.Init()
	instrumentInit()
	buildModeInit()
	if err := fsys.Init(base.Cwd()); err != nil {
		base.Fatalf("go: %v", err)
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
}

// fuzzInstrumentFlags returns compiler flags that enable fuzzing instrumation
// on supported platforms.
//
// On unsupported platforms, fuzzInstrumentFlags returns nil, meaning no
// instrumentation is added. 'go test -fuzz' still works without coverage,
// but it generates random inputs without guidance, so it's much less effective.
func fuzzInstrumentFlags() []string {
	if !sys.FuzzInstrumented(cfg.Goos, cfg.Goarch) {
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
	if cfg.BuildMSan && !sys.MSanSupported(cfg.Goos, cfg.Goarch) {
		fmt.Fprintf(os.Stderr, "-msan is not supported on %s/%s\n", cfg.Goos, cfg.Goarch)
		base.SetExitStatus(2)
		base.Exit()
	}
	if cfg.BuildRace && !sys.RaceDetectorSupported(cfg.Goos, cfg.Goarch) {
		fmt.Fprintf(os.Stderr, "-race is not supported on %s/%s\n", cfg.Goos, cfg.Goarch)
		base.SetExitStatus(2)
		base.Exit()
	}
	if cfg.BuildASan && !sys.ASanSupported(cfg.Goos, cfg.Goarch) {
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
		// MSAN does not support non-PIE binaries on ARM64.
		// See issue #33712 for details.
		if cfg.Goos == "linux" && cfg.Goarch == "arm64" && cfg.BuildBuildmode == "default" {
			cfg.BuildBuildmode = "pie"
		}
	}
	if cfg.BuildASan {
		mode = "asan"
	}
	modeFlag := "-" + mode

	if !cfg.BuildContext.CgoEnabled {
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
		switch cfg.Goos {
		case "android":
			codegenArg = "-shared"
			ldBuildmode = "pie"
		case "windows":
			if cfg.BuildRace {
				ldBuildmode = "exe"
			} else {
				ldBuildmode = "pie"
			}
		case "ios":
			codegenArg = "-shared"
			ldBuildmode = "pie"
		case "darwin":
			switch cfg.Goarch {
			case "arm64":
				codegenArg = "-shared"
			}
			fallthrough
		default:
			ldBuildmode = "exe"
		}
		if gccgo {
			codegenArg = ""
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
		if cfg.BuildRace {
			base.Fatalf("-buildmode=pie not supported when -race is enabled")
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

	if !sys.BuildModeSupported(cfg.BuildToolchainName, cfg.BuildBuildmode, cfg.Goos, cfg.Goarch) {
		base.Fatalf("-buildmode=%s not supported on %s/%s\n", cfg.BuildBuildmode, cfg.Goos, cfg.Goarch)
	}

	if cfg.BuildLinkshared {
		if !sys.BuildModeSupported(cfg.BuildToolchainName, "shared", cfg.Goos, cfg.Goarch) {
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
			out, err := exec.Command(cc, "--version").Output()
			if err != nil {
				// Compiler does not support "--version" flag: not Clang or GCC.
				return err
			}

			var match [][]byte
			if bytes.HasPrefix(out, []byte("gcc")) {
				compiler.name = "gcc"
				out, err := exec.Command(cc, "-v").CombinedOutput()
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
// misc/cgo/testsanitizers/cc_test.go
// compilerRequiredAsanVersion reports whether the compiler is the version
// required by Asan.
func compilerRequiredAsanVersion() error {
	compiler, err := compilerVersion()
	if err != nil {
		return fmt.Errorf("-asan: the version of $(go env CC) could not be parsed")
	}

	switch compiler.name {
	case "gcc":
		if compiler.major < 7 {
			return fmt.Errorf("-asan is not supported with C compiler %d.%d\n", compiler.major, compiler.minor)
		}
	case "clang":
		if compiler.major < 9 {
			return fmt.Errorf("-asan is not supported with C compiler %d.%d\n", compiler.major, compiler.minor)
		}
	default:
		return fmt.Errorf("-asan: C compiler is not gcc or clang")
	}
	return nil
}
