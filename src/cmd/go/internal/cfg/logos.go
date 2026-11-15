// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfg

import (
	"fmt"
	"internal/logosfile"
	"os"
	"path/filepath"
	"strings"

	"cmd/go/internal/logos/settings"
	"cmd/internal/pathcache"
)

// Logos reports whether the current process was invoked as the logos
// command. The logos command is the same binary as the go command,
// selected by argv[0]: the base name of os.Args[0] with a ".exe"
// suffix removed must equal "logos". It is a package-level initializer,
// so defaultContext observes it when BuildContext is first built.
var Logos = strings.TrimSuffix(filepath.Base(os.Args[0]), ".exe") == "logos"

// WasmLoader is the text of the host loader a wasm build writes beside its
// output. cmd/go/internal/logos sets it at program start, because that
// package imports the packages that import work and so work cannot import
// it.
var WasmLoader string

// logosInitCC snapshots the user CC at process startup, covering both the
// process environment and the GOENV file. main.invoke later injects every
// MkEnv value (including a CC default) into the process environment, so a
// post-invoke Getenv can never observe an unset CC; the zig fallback and
// any other consumer must read this snapshot instead.
var logosInitCC = Getenv("CC")

// logosInitCXX snapshots the user CXX at process startup, covering both the
// process environment and the GOENV file, for the same reason as
// logosInitCC: the zig c++ fallback must tell an explicit user setting
// (including `go env -w CXX=...`) from an injected MkEnv default.
var logosInitCXX = Getenv("CXX")

// logosInitCGOEnabled snapshots the user CGO_ENABLED at process startup,
// covering both the process environment and the GOENV file. main.invoke
// injects the effective CGO_ENABLED into the process environment before
// commands run, so per-package precedence must read this snapshot instead
// of the live environment; see StartupCgoEnabled.
var logosInitCGOEnabled = Getenv("CGO_ENABLED")

var logosInitGOEXPERIMENT = RawGOEXPERIMENT

var logosInitGOAMD64 = Getenv("GOAMD64")

var logosInitGOARM64 = Getenv("GOARM64")

func LogosStartupEnv() settings.Env {
	return settings.Env{
		GOEXPERIMENT: logosInitGOEXPERIMENT,
		CGO_ENABLED:  logosInitCGOEnabled,
		GOAMD64:      logosInitGOAMD64,
		GOARM64:      logosInitGOARM64,
	}
}

// logosToolTags returns tags plus the "logos" tool tag in logos mode.
// defaultContext calls it, so every rebuilt BuildContext keeps the tag.
func logosToolTags(tags []string) []string {
	if !Logos {
		return tags
	}
	return append(tags, "logos")
}

// IsGoFile reports whether name is a Go-family source file: a ".go"
// file, or a loGo/s file when the current process is the logos command.
// It is valid only at run time, after init has set Logos.
func IsGoFile(name string) bool {
	if strings.HasSuffix(name, ".go") {
		return true
	}
	return Logos && logosfile.Is(name)
}

var logosZigTargets = map[string]string{
	"android/386":     "x86-linux-android",
	"android/amd64":   "x86_64-linux-android",
	"android/arm":     "arm-linux-androideabi",
	"android/arm64":   "aarch64-linux-android",
	"darwin/amd64":    "x86_64-macos",
	"darwin/arm64":    "aarch64-macos",
	"dragonfly/amd64": "x86_64-dragonfly",
	"freebsd/386":     "x86-freebsd",
	"freebsd/amd64":   "x86_64-freebsd",
	"freebsd/arm":     "arm-freebsd",
	"freebsd/arm64":   "aarch64-freebsd",
	"freebsd/riscv64": "riscv64-freebsd",
	"illumos/amd64":   "x86_64-illumos",
	"ios/amd64":       "x86_64-ios",
	"ios/arm64":       "aarch64-ios",
	"linux/386":       "x86-linux-musl",
	"linux/amd64":     "x86_64-linux-musl",
	"linux/arm":       "arm-linux-musl",
	"linux/arm64":     "aarch64-linux-musl",
	"linux/loong64":   "loongarch64-linux-musl",
	"linux/mips":      "mips-linux-musl",
	"linux/mips64":    "mips64-linux-muslabi64",
	"linux/mips64le":  "mips64el-linux-muslabi64",
	"linux/mipsle":    "mipsel-linux-musl",
	"linux/ppc64":     "powerpc64-linux-musl",
	"linux/ppc64le":   "powerpc64le-linux-musl",
	"linux/riscv64":   "riscv64-linux-musl",
	"linux/s390x":     "s390x-linux-musl",
	"linux/sparc64":   "sparc64-linux-musl",
	"netbsd/386":      "x86-netbsd",
	"netbsd/amd64":    "x86_64-netbsd",
	"netbsd/arm":      "arm-netbsd",
	"netbsd/arm64":    "aarch64-netbsd",
	"openbsd/386":     "x86-openbsd",
	"openbsd/amd64":   "x86_64-openbsd",
	"openbsd/arm":     "arm-openbsd",
	"openbsd/arm64":   "aarch64-openbsd",
	"openbsd/riscv64": "riscv64-openbsd",
	"windows/386":     "x86-windows-gnu",
	"windows/amd64":   "x86_64-windows-gnu",
	"windows/arm64":   "aarch64-windows-gnu",
}

func ZigTarget(goos, goarch string) (string, bool) {
	target, ok := logosZigTargets[goos+"/"+goarch]
	return target, ok
}

// StartupCgoEnabled reports the user CGO_ENABLED observed at process
// startup, before main.invoke injects the effective value into the process
// environment. Per-package cgo precedence must use this, not the live
// environment: after invoke the live value always looks explicitly set.
// It returns "" when the user set nothing; only "0" and "1" count as set.
func StartupCgoEnabled() string {
	return logosInitCGOEnabled
}

func ApplyLogosEnv(s *settings.Settings) {
	if !Logos {
		return
	}
	if s.Cgo.Opinion {
		BuildContext.CgoEnabled = s.Cgo.Enabled
	}
	CGOChanged = s.Cgo.Changed
	logosApplyRawExperiment(s.Experiment)
	RefreshLogosPreemptible()
}

// zigCompiler resolves one C-family compiler command for a cgo build given
// its current value: the zig fallback "zig <kind> -target <triple>" when
// cgo is on, the variable was unset in both the process environment and
// the GOENV file at startup, the current compiler is missing from PATH,
// and zig is on PATH; otherwise cmd unchanged. kind is "cc" for CC and
// "c++" for CXX. The startup snapshot tells an injected MkEnv default
// (which main.invoke writes into the process environment before builds
// run) from an explicit user setting, so only the former may fall back. It
// never writes the process environment, so "go env CC" and "go env CXX"
// keep showing the default or user compiler. It runs under both go and
// logos for Go and loGo/s packages alike. The triple follows the command
// target, which ForceHost already reset for host tool builds.
func zigCompiler(cmd, startup, kind string) string {
	if !BuildContext.CgoEnabled {
		return cmd
	}
	if startup != "" {
		return cmd
	}
	if compilerOnPath(cmd) {
		return cmd
	}
	if _, err := pathcache.LookPath("zig"); err != nil {
		return cmd
	}
	if target, ok := ZigTarget(Goos, Goarch); ok {
		return "zig " + kind + " -target " + target
	}
	fmt.Fprintf(os.Stderr, "go: zig compiler fallback unsupported for target %s/%s; set CC explicitly\n", Goos, Goarch)
	os.Exit(2)
	return ""
}

// compilerOnPath reports whether cmd is the compiler a cgo build would
// actually run under the current PATH. The injected default may be an
// absolute path baked in at toolchain build time; an absolute path is
// present only when its own directory is on PATH, so hiding the default
// compiler from PATH still enables the zig fallback.
func compilerOnPath(cmd string) bool {
	found, err := pathcache.LookPath(cmd)
	if err != nil {
		return false
	}
	if !filepath.IsAbs(cmd) {
		return true
	}
	viaPath, err := pathcache.LookPath(filepath.Base(cmd))
	if err != nil {
		return false
	}
	want, err1 := os.Stat(found)
	got, err2 := os.Stat(viaPath)
	if err1 != nil || err2 != nil {
		return false
	}
	return os.SameFile(want, got)
}

func ZigCC(cc string) string {
	return zigCompiler(cc, logosInitCC, "cc")
}

func ZigCXX(cxx string) string {
	return zigCompiler(cxx, logosInitCXX, "c++")
}
