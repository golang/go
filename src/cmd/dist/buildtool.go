// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Build toolchain using Go bootstrap version.
//
// The general strategy is to copy the source files we need into
// a new GOPATH workspace, adjust import paths appropriately,
// invoke the Go bootstrap toolchains go command to build those sources,
// and then copy the binaries back.

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// bootstrapDirs is a list of directories holding code that must be
// compiled with the Go bootstrap toolchain to produce the bootstrapTargets.
// All directories in this list are relative to and must be below $GOROOT/src.
//
// The list has two kinds of entries: names beginning with cmd/ with
// no other slashes, which are commands, and other paths, which are packages
// supporting the commands. Packages in the standard library can be listed
// if a newer copy needs to be substituted for the Go bootstrap copy when used
// by the command packages. Paths ending with /... automatically
// include all packages within subdirectories as well.
// These will be imported during bootstrap as bootstrap/name, like bootstrap/math/big.
var bootstrapDirs = []string{
	"cmp",
	"cmd/asm",
	"cmd/asm/internal/...",
	"cmd/cgo",
	"cmd/compile",
	"cmd/compile/internal/...",
	"cmd/internal/archive",
	"cmd/internal/bio",
	"cmd/internal/codesign",
	"cmd/internal/dwarf",
	"cmd/internal/edit",
	"cmd/internal/gcprog",
	"cmd/internal/goobj",
	"cmd/internal/notsha256",
	"cmd/internal/obj/...",
	"cmd/internal/objabi",
	"cmd/internal/pgo",
	"cmd/internal/pkgpath",
	"cmd/internal/quoted",
	"cmd/internal/src",
	"cmd/internal/sys",
	"cmd/internal/telemetry",
	"cmd/link",
	"cmd/link/internal/...",
	"compress/flate",
	"compress/zlib",
	"container/heap",
	"debug/dwarf",
	"debug/elf",
	"debug/macho",
	"debug/pe",
	"go/build/constraint",
	"go/constant",
	"go/version",
	"internal/abi",
	"internal/coverage",
	"cmd/internal/cov/covcmd",
	"internal/bisect",
	"internal/buildcfg",
	"internal/goarch",
	"internal/godebugs",
	"internal/goexperiment",
	"internal/goroot",
	"internal/gover",
	"internal/goversion",
	// internal/lazyregexp is provided by Go 1.17, which permits it to
	// be imported by other packages in this list, but is not provided
	// by the Go 1.17 version of gccgo. It's on this list only to
	// support gccgo, and can be removed if we require gccgo 14 or later.
	"internal/lazyregexp",
	"internal/pkgbits",
	"internal/platform",
	"internal/profile",
	"internal/race",
	"internal/saferio",
	"internal/syscall/unix",
	"internal/types/errors",
	"internal/unsafeheader",
	"internal/xcoff",
	"internal/zstd",
	"math/bits",
	"sort",
}

// File prefixes that are ignored by go/build anyway, and cause
// problems with editor generated temporary files (#18931).
var ignorePrefixes = []string{
	".",
	"_",
	"#",
}

// File suffixes that use build tags introduced since Go 1.17.
// These must not be copied into the bootstrap build directory.
// Also ignore test files.
var ignoreSuffixes = []string{
	"_test.s",
	"_test.go",
	// Skip PGO profile. No need to build toolchain1 compiler
	// with PGO. And as it is not a text file the import path
	// rewrite will break it.
	".pgo",
}

var tryDirs = []string{
	"sdk/go1.17",
	"go1.17",
}

func bootstrapBuildTools() {
	goroot_bootstrap := os.Getenv("GOROOT_BOOTSTRAP")
	if goroot_bootstrap == "" {
		home := os.Getenv("HOME")
		goroot_bootstrap = pathf("%s/go1.4", home)
		for _, d := range tryDirs {
			if p := pathf("%s/%s", home, d); isdir(p) {
				goroot_bootstrap = p
			}
		}
	}
	xprintf("Building Go toolchain1 using %s.\n", goroot_bootstrap)

	mkbuildcfg(pathf("%s/src/internal/buildcfg/zbootstrap.go", goroot))
	mkobjabi(pathf("%s/src/cmd/internal/objabi/zbootstrap.go", goroot))

	// Use $GOROOT/pkg/bootstrap as the bootstrap workspace root.
	// We use a subdirectory of $GOROOT/pkg because that's the
	// space within $GOROOT where we store all generated objects.
	// We could use a temporary directory outside $GOROOT instead,
	// but it is easier to debug on failure if the files are in a known location.
	workspace := pathf("%s/pkg/bootstrap", goroot)
	xremoveall(workspace)
	xatexit(func() { xremoveall(workspace) })
	base := pathf("%s/src/bootstrap", workspace)
	xmkdirall(base)

	// Copy source code into $GOROOT/pkg/bootstrap and rewrite import paths.
	writefile("module bootstrap\ngo 1.20\n", pathf("%s/%s", base, "go.mod"), 0)
	for _, dir := range bootstrapDirs {
		recurse := strings.HasSuffix(dir, "/...")
		dir = strings.TrimSuffix(dir, "/...")
		filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				fatalf("walking bootstrap dirs failed: %v: %v", path, err)
			}

			name := filepath.Base(path)
			src := pathf("%s/src/%s", goroot, path)
			dst := pathf("%s/%s", base, path)

			if info.IsDir() {
				if !recurse && path != dir || name == "testdata" {
					return filepath.SkipDir
				}

				xmkdirall(dst)
				if path == "cmd/cgo" {
					// Write to src because we need the file both for bootstrap
					// and for later in the main build.
					mkzdefaultcc("", pathf("%s/zdefaultcc.go", src))
					mkzdefaultcc("", pathf("%s/zdefaultcc.go", dst))
				}
				return nil
			}

			for _, pre := range ignorePrefixes {
				if strings.HasPrefix(name, pre) {
					return nil
				}
			}
			for _, suf := range ignoreSuffixes {
				if strings.HasSuffix(name, suf) {
					return nil
				}
			}

			text := bootstrapRewriteFile(src)
			writefile(text, dst, 0)
			return nil
		})
	}

	// Set up environment for invoking Go bootstrap toolchains go command.
	// GOROOT points at Go bootstrap GOROOT,
	// GOPATH points at our bootstrap workspace,
	// GOBIN is empty, so that binaries are installed to GOPATH/bin,
	// and GOOS, GOHOSTOS, GOARCH, and GOHOSTOS are empty,
	// so that Go bootstrap toolchain builds whatever kind of binary it knows how to build.
	// Restore GOROOT, GOPATH, and GOBIN when done.
	// Don't bother with GOOS, GOHOSTOS, GOARCH, and GOHOSTARCH,
	// because setup will take care of those when bootstrapBuildTools returns.

	defer os.Setenv("GOROOT", os.Getenv("GOROOT"))
	os.Setenv("GOROOT", goroot_bootstrap)

	defer os.Setenv("GOPATH", os.Getenv("GOPATH"))
	os.Setenv("GOPATH", workspace)

	defer os.Setenv("GOBIN", os.Getenv("GOBIN"))
	os.Setenv("GOBIN", "")

	os.Setenv("GOOS", "")
	os.Setenv("GOHOSTOS", "")
	os.Setenv("GOARCH", "")
	os.Setenv("GOHOSTARCH", "")

	// Run Go bootstrap to build binaries.
	// Use the math_big_pure_go build tag to disable the assembly in math/big
	// which may contain unsupported instructions.
	// Use the purego build tag to disable other assembly code,
	// such as in cmd/internal/notsha256.
	cmd := []string{
		pathf("%s/bin/go", goroot_bootstrap),
		"install",
		"-tags=math_big_pure_go compiler_bootstrap purego",
	}
	if vflag > 0 {
		cmd = append(cmd, "-v")
	}
	if tool := os.Getenv("GOBOOTSTRAP_TOOLEXEC"); tool != "" {
		cmd = append(cmd, "-toolexec="+tool)
	}
	cmd = append(cmd, "bootstrap/cmd/...")
	run(base, ShowOutput|CheckExit, cmd...)

	// Copy binaries into tool binary directory.
	for _, name := range bootstrapDirs {
		if !strings.HasPrefix(name, "cmd/") {
			continue
		}
		name = name[len("cmd/"):]
		if !strings.Contains(name, "/") {
			copyfile(pathf("%s/%s%s", tooldir, name, exe), pathf("%s/bin/%s%s", workspace, name, exe), writeExec)
		}
	}

	if vflag > 0 {
		xprintf("\n")
	}
}

var ssaRewriteFileSubstring = filepath.FromSlash("src/cmd/compile/internal/ssa/rewrite")

// isUnneededSSARewriteFile reports whether srcFile is a
// src/cmd/compile/internal/ssa/rewriteARCHNAME.go file for an
// architecture that isn't for the given GOARCH.
//
// When unneeded is true archCaps is the rewrite base filename without
// the "rewrite" prefix or ".go" suffix: AMD64, 386, ARM, ARM64, etc.
func isUnneededSSARewriteFile(srcFile, goArch string) (archCaps string, unneeded bool) {
	if !strings.Contains(srcFile, ssaRewriteFileSubstring) {
		return "", false
	}
	fileArch := strings.TrimSuffix(strings.TrimPrefix(filepath.Base(srcFile), "rewrite"), ".go")
	if fileArch == "" {
		return "", false
	}
	b := fileArch[0]
	if b == '_' || ('a' <= b && b <= 'z') {
		return "", false
	}
	archCaps = fileArch
	fileArch = strings.ToLower(fileArch)
	fileArch = strings.TrimSuffix(fileArch, "splitload")
	fileArch = strings.TrimSuffix(fileArch, "latelower")
	if fileArch == goArch {
		return "", false
	}
	if fileArch == strings.TrimSuffix(goArch, "le") {
		return "", false
	}
	return archCaps, true
}

func bootstrapRewriteFile(srcFile string) string {
	// During bootstrap, generate dummy rewrite files for
	// irrelevant architectures. We only need to build a bootstrap
	// binary that works for the current gohostarch.
	// This saves 6+ seconds of bootstrap.
	if archCaps, ok := isUnneededSSARewriteFile(srcFile, gohostarch); ok {
		return fmt.Sprintf(`%spackage ssa

func rewriteValue%s(v *Value) bool { panic("unused during bootstrap") }
func rewriteBlock%s(b *Block) bool { panic("unused during bootstrap") }
`, generatedHeader, archCaps, archCaps)
	}

	return bootstrapFixImports(srcFile)
}

var (
	importRE      = regexp.MustCompile(`\Aimport\s+(\.|[A-Za-z0-9_]+)?\s*"([^"]+)"\s*(//.*)?\n\z`)
	importBlockRE = regexp.MustCompile(`\A\s*(?:(\.|[A-Za-z0-9_]+)?\s*"([^"]+)")?\s*(//.*)?\n\z`)
)

func bootstrapFixImports(srcFile string) string {
	text := readfile(srcFile)
	if !strings.Contains(srcFile, "/cmd/") && !strings.Contains(srcFile, `\cmd\`) {
		text = regexp.MustCompile(`\bany\b`).ReplaceAllString(text, "interface{}")
	}
	lines := strings.SplitAfter(text, "\n")
	inBlock := false
	inComment := false
	for i, line := range lines {
		if strings.HasSuffix(line, "*/\n") {
			inComment = false
		}
		if strings.HasSuffix(line, "/*\n") {
			inComment = true
		}
		if inComment {
			continue
		}
		if strings.HasPrefix(line, "import (") {
			inBlock = true
			continue
		}
		if inBlock && strings.HasPrefix(line, ")") {
			inBlock = false
			continue
		}

		var m []string
		if !inBlock {
			if !strings.HasPrefix(line, "import ") {
				continue
			}
			m = importRE.FindStringSubmatch(line)
			if m == nil {
				fatalf("%s:%d: invalid import declaration: %q", srcFile, i+1, line)
			}
		} else {
			m = importBlockRE.FindStringSubmatch(line)
			if m == nil {
				fatalf("%s:%d: invalid import block line", srcFile, i+1)
			}
			if m[2] == "" {
				continue
			}
		}

		path := m[2]
		if strings.HasPrefix(path, "cmd/") {
			path = "bootstrap/" + path
		} else {
			for _, dir := range bootstrapDirs {
				if path == dir {
					path = "bootstrap/" + dir
					break
				}
			}
		}

		// Rewrite use of internal/reflectlite to be plain reflect.
		if path == "internal/reflectlite" {
			lines[i] = strings.ReplaceAll(line, `"reflect"`, `reflectlite "reflect"`)
			continue
		}

		// Otherwise, reject direct imports of internal packages,
		// since that implies knowledge of internal details that might
		// change from one bootstrap toolchain to the next.
		// There are many internal packages that are listed in
		// bootstrapDirs and made into bootstrap copies based on the
		// current repo's source code. Those are fine; this is catching
		// references to internal packages in the older bootstrap toolchain.
		if strings.HasPrefix(path, "internal/") {
			fatalf("%s:%d: bootstrap-copied source file cannot import %s", srcFile, i+1, path)
		}
		if path != m[2] {
			lines[i] = strings.ReplaceAll(line, `"`+m[2]+`"`, `"`+path+`"`)
		}
	}

	lines[0] = generatedHeader + "// This is a bootstrap copy of " + srcFile + "\n\n//line " + srcFile + ":1\n" + lines[0]

	return strings.Join(lines, "")
}
