// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cfg holds configuration shared by multiple parts
// of the go command.
package cfg

import (
	"fmt"
	"go/build"
	"os"
	"path/filepath"
	"runtime"

	"cmd/internal/objabi"
)

// These are general "build flags" used by build and other commands.
var (
	BuildA                 bool   // -a flag
	BuildBuildmode         string // -buildmode flag
	BuildContext           = defaultContext()
	BuildMod               string             // -mod flag
	BuildI                 bool               // -i flag
	BuildLinkshared        bool               // -linkshared flag
	BuildMSan              bool               // -msan flag
	BuildN                 bool               // -n flag
	BuildO                 string             // -o flag
	BuildP                 = runtime.NumCPU() // -p flag
	BuildPkgdir            string             // -pkgdir flag
	BuildRace              bool               // -race flag
	BuildToolexec          []string           // -toolexec flag
	BuildToolchainName     string
	BuildToolchainCompiler func() string
	BuildToolchainLinker   func() string
	BuildV                 bool // -v flag
	BuildWork              bool // -work flag
	BuildX                 bool // -x flag

	CmdName string // "build", "install", "list", etc.

	DebugActiongraph string // -debug-actiongraph flag (undocumented, unstable)
)

func defaultContext() build.Context {
	ctxt := build.Default
	ctxt.JoinPath = filepath.Join // back door to say "do not use go command"
	return ctxt
}

func init() {
	BuildToolchainCompiler = func() string { return "missing-compiler" }
	BuildToolchainLinker = func() string { return "missing-linker" }
}

// An EnvVar is an environment variable Name=Value.
type EnvVar struct {
	Name  string
	Value string
}

// OrigEnv is the original environment of the program at startup.
var OrigEnv []string

// CmdEnv is the new environment for running go tool commands.
// User binaries (during go test or go run) are run with OrigEnv,
// not CmdEnv.
var CmdEnv []EnvVar

// Global build parameters (used during package load)
var (
	Goarch    = BuildContext.GOARCH
	Goos      = BuildContext.GOOS
	ExeSuffix string
	Gopath    = filepath.SplitList(BuildContext.GOPATH)

	// ModulesEnabled specifies whether the go command is running
	// in module-aware mode (as opposed to GOPATH mode).
	// It is equal to modload.Enabled, but not all packages can import modload.
	ModulesEnabled bool

	// GoModInGOPATH records whether we've found a go.mod in GOPATH/src
	// in GO111MODULE=auto mode. In that case, we don't use modules
	// but people might expect us to, so 'go get' warns.
	GoModInGOPATH string
)

func init() {
	if Goos == "windows" {
		ExeSuffix = ".exe"
	}
}

var (
	GOROOT       = findGOROOT()
	GOBIN        = os.Getenv("GOBIN")
	GOROOTbin    = filepath.Join(GOROOT, "bin")
	GOROOTpkg    = filepath.Join(GOROOT, "pkg")
	GOROOTsrc    = filepath.Join(GOROOT, "src")
	GOROOT_FINAL = findGOROOT_FINAL()

	// Used in envcmd.MkEnv and build ID computations.
	GOARM    = fmt.Sprint(objabi.GOARM)
	GO386    = objabi.GO386
	GOMIPS   = objabi.GOMIPS
	GOMIPS64 = objabi.GOMIPS64
)

// Update build context to use our computed GOROOT.
func init() {
	BuildContext.GOROOT = GOROOT
	if runtime.Compiler != "gccgo" {
		// Note that we must use runtime.GOOS and runtime.GOARCH here,
		// as the tool directory does not move based on environment
		// variables. This matches the initialization of ToolDir in
		// go/build, except for using GOROOT rather than
		// runtime.GOROOT.
		build.ToolDir = filepath.Join(GOROOT, "pkg/tool/"+runtime.GOOS+"_"+runtime.GOARCH)
	}
}

// There is a copy of findGOROOT, isSameDir, and isGOROOT in
// x/tools/cmd/godoc/goroot.go.
// Try to keep them in sync for now.

// findGOROOT returns the GOROOT value, using either an explicitly
// provided environment variable, a GOROOT that contains the current
// os.Executable value, or else the GOROOT that the binary was built
// with from runtime.GOROOT().
//
// There is a copy of this code in x/tools/cmd/godoc/goroot.go.
func findGOROOT() string {
	if env := os.Getenv("GOROOT"); env != "" {
		return filepath.Clean(env)
	}
	def := filepath.Clean(runtime.GOROOT())
	if runtime.Compiler == "gccgo" {
		// gccgo has no real GOROOT, and it certainly doesn't
		// depend on the executable's location.
		return def
	}
	exe, err := os.Executable()
	if err == nil {
		exe, err = filepath.Abs(exe)
		if err == nil {
			if dir := filepath.Join(exe, "../.."); isGOROOT(dir) {
				// If def (runtime.GOROOT()) and dir are the same
				// directory, prefer the spelling used in def.
				if isSameDir(def, dir) {
					return def
				}
				return dir
			}
			exe, err = filepath.EvalSymlinks(exe)
			if err == nil {
				if dir := filepath.Join(exe, "../.."); isGOROOT(dir) {
					if isSameDir(def, dir) {
						return def
					}
					return dir
				}
			}
		}
	}
	return def
}

func findGOROOT_FINAL() string {
	def := GOROOT
	if env := os.Getenv("GOROOT_FINAL"); env != "" {
		def = filepath.Clean(env)
	}
	return def
}

// isSameDir reports whether dir1 and dir2 are the same directory.
func isSameDir(dir1, dir2 string) bool {
	if dir1 == dir2 {
		return true
	}
	info1, err1 := os.Stat(dir1)
	info2, err2 := os.Stat(dir2)
	return err1 == nil && err2 == nil && os.SameFile(info1, info2)
}

// isGOROOT reports whether path looks like a GOROOT.
//
// It does this by looking for the path/pkg/tool directory,
// which is necessary for useful operation of the cmd/go tool,
// and is not typically present in a GOPATH.
//
// There is a copy of this code in x/tools/cmd/godoc/goroot.go.
func isGOROOT(path string) bool {
	stat, err := os.Stat(filepath.Join(path, "pkg", "tool"))
	if err != nil {
		return false
	}
	return stat.IsDir()
}
