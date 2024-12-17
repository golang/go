// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fips implements support for the GOFIPS140 build setting.
//
// The GOFIPS140 build setting controls two aspects of the build:
//
//   - Whether binaries are built to default to running in FIPS-140 mode,
//     meaning whether they default to GODEBUG=fips140=on or =off.
//
//   - Which copy of the crypto/internal/fips140 source code to use.
//     The default is obviously GOROOT/src/crypto/internal/fips140,
//     but earlier snapshots that have differing levels of external
//     validation and certification are stored in GOROOT/lib/fips140
//     and can be substituted into the build instead.
//
// This package provides the logic needed by the rest of the go command
// to make those decisions and implement the resulting policy.
//
// [Init] must be called to initialize the FIPS logic. It may fail and
// call base.Fatalf.
//
// When GOFIPS140=off, [Enabled] returns false, and the build is
// unchanged from its usual behaviors.
//
// When GOFIPS140 is anything else, [Enabled] returns true, and the build
// sets the default GODEBUG to include fips140=on. This will make
// binaries change their behavior at runtime to confirm to various
// FIPS-140 details. [cmd/go/internal/load.defaultGODEBUG] calls
// [fips.Enabled] when preparing the default settings.
//
// For all builds, FIPS code and data is laid out in contiguous regions
// that are conceptually concatenated into a "fips object file" that the
// linker hashes and then binaries can re-hash at startup to detect
// corruption of those symbols. When [Enabled] is true, the link step
// passes -fipso={a.Objdir}/fips.o to the linker to save a copy of the
// fips.o file. Since the first build target always uses a.Objdir set to
// $WORK/b001, a build like
//
//	GOFIPS140=latest go build -work my/binary
//
// will leave fips.o behind in $WORK/b001. Auditors like to be able to
// see that file. Accordingly, when [Enabled] returns true,
// [cmd/go/internal/work.Builder.useCache] arranges never to cache linker
// output, so that the link step always runs, and fips.o is always left
// behind in the link step. If this proves too slow, we could always
// cache fips.o as an extra link output and then restore it when -work is
// set, but we went a very long time never caching link steps at all, so
// not caching them in FIPS mode seems perfectly fine.
//
// When GOFIPS140 is set to something besides off and latest, [Snapshot]
// returns true, indicating that the build should replace the latest copy
// of crypto/internal/fips140 with an earlier snapshot. The reason to do
// this is to use a copy that has been through additional lab validation
// (an "in-process" module) or NIST certification (a "certified" module).
// The snapshots are stored in GOROOT/lib/fips140 in module zip form.
// When a snapshot is being used, Init unpacks it into the module cache
// and then uses that directory as the source location.
//
// A FIPS snapshot like v1.2.3 is integrated into the build in two different ways.
//
// First, the snapshot's fips140 directory replaces crypto/internal/fips140
// using fsys.Bind. The effect is to appear to have deleted crypto/internal/fips140
// and everything below it, replacing it with the single subdirectory
// crypto/internal/fips140/v1.2.3, which now has the FIPS packages.
// This virtual file system replacement makes patterns like std and crypto...
// automatically see the snapshot packages instead of the original packages
// as they walk GOROOT/src/crypto/internal/fips140.
//
// Second, ResolveImport is called to resolve an import like crypto/internal/fips140/sha256.
// When snapshot v1.2.3 is being used, ResolveImport translates that path to
// crypto/internal/fips140/v1.2.3/sha256 and returns the actual source directory
// in the unpacked snapshot. Using the actual directory instead of the
// virtual directory GOROOT/src/crypto/internal/fips140/v1.2.3 makes sure
// that other tools using go list -json output can find the sources,
// as well as making sure builds have a real directory in which to run the
// assembler, compiler, and so on. The translation of the import path happens
// in the same code that handles mapping golang.org/x/mod to
// cmd/vendor/golang.org/x/mod when building commands.
//
// It is not strictly required to include v1.2.3 in the import path when using
// a snapshot - we could make things work without doing that - but including
// the v1.2.3 gives a different version of the code a different name, which is
// always a good general rule. In particular, it will mean that govulncheck need
// not have any special cases for crypto/internal/fips140 at all. The reports simply
// need to list the relevant symbols in a given Go version. (For example, if a bug
// is only in the in-tree copy but not the snapshots, it doesn't list the snapshot
// symbols; if it's in any snapshots, it has to list the specific snapshot symbols
// in addition to the “normal” symbol.)
package fips140

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/str"
	"context"
	"os"
	"path"
	"path/filepath"
	"strings"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

// Init initializes the FIPS settings.
// It must be called before using any other functions in this package.
// If initialization fails, Init calls base.Fatalf.
func Init() {
	if initDone {
		return
	}
	initDone = true
	initVersion()
	initDir()
	if Snapshot() {
		fsys.Bind(Dir(), filepath.Join(cfg.GOROOT, "src/crypto/internal/fips140"))
	}

	if cfg.Experiment.BoringCrypto && Enabled() {
		base.Fatalf("go: cannot use GOFIPS140 with GOEXPERIMENT=boringcrypto")
	}
}

var initDone bool

// checkInit panics if Init has not been called.
func checkInit() {
	if !initDone {
		panic("fips: not initialized")
	}
}

// Version reports the GOFIPS140 version in use,
// which is either "off", "latest", or a version like "v1.2.3".
// If GOFIPS140 is set to an alias like "inprocess" or "certified",
// Version returns the underlying version.
func Version() string {
	checkInit()
	return version
}

// Enabled reports whether FIPS mode is enabled at all.
// That is, it reports whether GOFIPS140 is set to something besides "off".
func Enabled() bool {
	checkInit()
	return version != "off"
}

// Snapshot reports whether FIPS mode is using a source snapshot
// rather than $GOROOT/src/crypto/internal/fips140.
// That is, it reports whether GOFIPS140 is set to something besides "latest" or "off".
func Snapshot() bool {
	checkInit()
	return version != "latest" && version != "off"
}

var version string

func initVersion() {
	// For off and latest, use the local source tree.
	v := cfg.GOFIPS140
	if v == "off" || v == "" {
		version = "off"
		return
	}
	if v == "latest" {
		version = "latest"
		return
	}

	// Otherwise version must exist in lib/fips140, either as
	// a .zip (a source snapshot like v1.2.0.zip)
	// or a .txt (a redirect like inprocess.txt, containing a version number).
	if strings.Contains(v, "/") || strings.Contains(v, `\`) || strings.Contains(v, "..") {
		base.Fatalf("go: malformed GOFIPS140 version %q", cfg.GOFIPS140)
	}
	if cfg.GOROOT == "" {
		base.Fatalf("go: missing GOROOT for GOFIPS140")
	}

	file := filepath.Join(cfg.GOROOT, "lib", "fips140", v)
	if data, err := os.ReadFile(file + ".txt"); err == nil {
		v = strings.TrimSpace(string(data))
		file = filepath.Join(cfg.GOROOT, "lib", "fips140", v)
		if _, err := os.Stat(file + ".zip"); err != nil {
			base.Fatalf("go: unknown GOFIPS140 version %q (from %q)", v, cfg.GOFIPS140)
		}
	}

	if _, err := os.Stat(file + ".zip"); err == nil {
		// Found version. Add a build tag.
		cfg.BuildContext.BuildTags = append(cfg.BuildContext.BuildTags, "fips140"+semver.MajorMinor(v))
		version = v
		return
	}

	base.Fatalf("go: unknown GOFIPS140 version %q", v)
}

// Dir reports the directory containing the crypto/internal/fips140 source code.
// If Snapshot() is false, Dir returns GOROOT/src/crypto/internal/fips140.
// Otherwise Dir ensures that the snapshot has been unpacked into the
// module cache and then returns the directory in the module cache
// corresponding to the crypto/internal/fips140 directory.
func Dir() string {
	checkInit()
	return dir
}

var dir string

func initDir() {
	v := version
	if v == "latest" || v == "off" {
		dir = filepath.Join(cfg.GOROOT, "src/crypto/internal/fips140")
		return
	}

	mod := module.Version{Path: "golang.org/fips140", Version: v}
	file := filepath.Join(cfg.GOROOT, "lib/fips140", v+".zip")
	zdir, err := modfetch.Unzip(context.Background(), mod, file)
	if err != nil {
		base.Fatalf("go: unpacking GOFIPS140=%v: %v", v, err)
	}
	dir = filepath.Join(zdir, "fips140")
	return
}

// ResolveImport resolves the import path imp.
// If it is of the form crypto/internal/fips140/foo
// (not crypto/internal/fips140/v1.2.3/foo)
// and we are using a snapshot, then LookupImport
// rewrites the path to crypto/internal/fips140/v1.2.3/foo
// and returns that path and its location in the unpacked
// FIPS snapshot.
func ResolveImport(imp string) (newPath, dir string, ok bool) {
	checkInit()
	const fips = "crypto/internal/fips140"
	if !Snapshot() || !str.HasPathPrefix(imp, fips) {
		return "", "", false
	}
	fipsv := path.Join(fips, version)
	var sub string
	if str.HasPathPrefix(imp, fipsv) {
		sub = "." + imp[len(fipsv):]
	} else {
		sub = "." + imp[len(fips):]
	}
	newPath = path.Join(fips, version, sub)
	dir = filepath.Join(Dir(), version, sub)
	return newPath, dir, true
}
