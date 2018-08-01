// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Build initialization (after flag parsing).

package work

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func BuildInit() {
	load.ModInit()
	instrumentInit()
	buildModeInit()

	// Make sure -pkgdir is absolute, because we run commands
	// in different directories.
	if cfg.BuildPkgdir != "" && !filepath.IsAbs(cfg.BuildPkgdir) {
		p, err := filepath.Abs(cfg.BuildPkgdir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "go %s: evaluating -pkgdir: %v\n", flag.Args()[0], err)
			os.Exit(2)
		}
		cfg.BuildPkgdir = p
	}
}

func instrumentInit() {
	if !cfg.BuildRace && !cfg.BuildMSan {
		return
	}
	if cfg.BuildRace && cfg.BuildMSan {
		fmt.Fprintf(os.Stderr, "go %s: may not use -race and -msan simultaneously\n", flag.Args()[0])
		os.Exit(2)
	}
	if cfg.BuildMSan && (cfg.Goos != "linux" || cfg.Goarch != "amd64" && cfg.Goarch != "arm64") {
		fmt.Fprintf(os.Stderr, "-msan is not supported on %s/%s\n", cfg.Goos, cfg.Goarch)
		os.Exit(2)
	}
	if cfg.BuildRace {
		platform := cfg.Goos + "/" + cfg.Goarch
		switch platform {
		default:
			fmt.Fprintf(os.Stderr, "go %s: -race is only supported on linux/amd64, linux/ppc64le, freebsd/amd64, netbsd/amd64, darwin/amd64 and windows/amd64\n", flag.Args()[0])
			os.Exit(2)
		case "linux/amd64", "linux/ppc64le", "freebsd/amd64", "netbsd/amd64", "darwin/amd64", "windows/amd64":
			// race supported on these platforms
		}
	}
	mode := "race"
	if cfg.BuildMSan {
		mode = "msan"
	}
	modeFlag := "-" + mode

	if !cfg.BuildContext.CgoEnabled {
		fmt.Fprintf(os.Stderr, "go %s: %s requires cgo; enable cgo by setting CGO_ENABLED=1\n", flag.Args()[0], modeFlag)
		os.Exit(2)
	}
	forcedGcflags = append(forcedGcflags, modeFlag)
	forcedLdflags = append(forcedLdflags, modeFlag)

	if cfg.BuildContext.InstallSuffix != "" {
		cfg.BuildContext.InstallSuffix += "_"
	}
	cfg.BuildContext.InstallSuffix += mode
	cfg.BuildContext.BuildTags = append(cfg.BuildContext.BuildTags, mode)
}

func buildModeInit() {
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
				if platform == "linux/ppc64" {
					base.Fatalf("-buildmode=c-archive not supported on %s\n", platform)
				}
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
			case "linux/amd64", "linux/arm", "linux/arm64", "linux/386", "linux/ppc64le", "linux/s390x",
				"android/amd64", "android/arm", "android/arm64", "android/386",
				"freebsd/amd64":
				codegenArg = "-shared"
			case "darwin/amd64", "darwin/386":
			case "windows/amd64", "windows/386":
				// Do not add usual .exe suffix to the .dll file.
				cfg.ExeSuffix = ""
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
			base.Fatalf("-buildmode=pie not supported by gccgo")
		} else {
			switch platform {
			case "linux/386", "linux/amd64", "linux/arm", "linux/arm64", "linux/ppc64le", "linux/s390x",
				"android/amd64", "android/arm", "android/arm64", "android/386",
				"freebsd/amd64":
				codegenArg = "-shared"
			case "darwin/amd64":
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
			case "linux/amd64", "linux/arm", "linux/arm64", "linux/386", "linux/s390x", "linux/ppc64le",
				"android/amd64", "android/arm", "android/arm64", "android/386":
			case "darwin/amd64":
				// Skip DWARF generation due to #21647
				forcedLdflags = append(forcedLdflags, "-w")
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
				forcedAsmflags = append(forcedAsmflags, "-D=GOBUILDMODE_shared=1")
			default:
				base.Fatalf("-linkshared not supported on %s\n", platform)
			}
			codegenArg = "-dynlink"
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
		// ok
	case "readonly", "vendor":
		if load.ModLookup == nil && !inGOFLAGS("-mod") {
			base.Fatalf("build flag -mod=%s only valid when using modules", cfg.BuildMod)
		}
	default:
		base.Fatalf("-mod=%s not supported (can be '', 'readonly', or 'vendor')", cfg.BuildMod)
	}
}

func inGOFLAGS(flag string) bool {
	for _, goflag := range base.GOFLAGS() {
		name := goflag
		if strings.HasPrefix(name, "--") {
			name = name[1:]
		}
		if i := strings.Index(name, "="); i >= 0 {
			name = name[:i]
		}
		if name == flag {
			return true
		}
	}
	return false
}
