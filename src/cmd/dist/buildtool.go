// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Build toolchain using Go 1.4.
//
// The general strategy is to copy the source files we need into
// a new GOPATH workspace, adjust import paths appropriately,
// invoke the Go 1.4 go command to build those sources,
// and then copy the binaries back.

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// bootstrapDirs is a list of directories holding code that must be
// compiled with a Go 1.4 toolchain to produce the bootstrapTargets.
// All directories in this list are relative to and must be below $GOROOT/src.
//
// The list has have two kinds of entries: names beginning with cmd/ with
// no other slashes, which are commands, and other paths, which are packages
// supporting the commands. Packages in the standard library can be listed
// if a newer copy needs to be substituted for the Go 1.4 copy when used
// by the command packages.
// These will be imported during bootstrap as bootstrap/name, like bootstrap/math/big.
var bootstrapDirs = []string{
	"cmd/asm",
	"cmd/asm/internal/arch",
	"cmd/asm/internal/asm",
	"cmd/asm/internal/flags",
	"cmd/asm/internal/lex",
	"cmd/cgo",
	"cmd/compile",
	"cmd/compile/internal/amd64",
	"cmd/compile/internal/arm",
	"cmd/compile/internal/arm64",
	"cmd/compile/internal/gc",
	"cmd/compile/internal/logopt",
	"cmd/compile/internal/mips",
	"cmd/compile/internal/mips64",
	"cmd/compile/internal/ppc64",
	"cmd/compile/internal/riscv64",
	"cmd/compile/internal/s390x",
	"cmd/compile/internal/ssa",
	"cmd/compile/internal/syntax",
	"cmd/compile/internal/types",
	"cmd/compile/internal/x86",
	"cmd/compile/internal/wasm",
	"cmd/internal/bio",
	"cmd/internal/gcprog",
	"cmd/internal/dwarf",
	"cmd/internal/edit",
	"cmd/internal/goobj2",
	"cmd/internal/objabi",
	"cmd/internal/obj",
	"cmd/internal/obj/arm",
	"cmd/internal/obj/arm64",
	"cmd/internal/obj/mips",
	"cmd/internal/obj/ppc64",
	"cmd/internal/obj/riscv",
	"cmd/internal/obj/s390x",
	"cmd/internal/obj/x86",
	"cmd/internal/obj/wasm",
	"cmd/internal/src",
	"cmd/internal/sys",
	"cmd/link",
	"cmd/link/internal/amd64",
	"cmd/link/internal/arm",
	"cmd/link/internal/arm64",
	"cmd/link/internal/ld",
	"cmd/link/internal/loadelf",
	"cmd/link/internal/loader",
	"cmd/link/internal/loadmacho",
	"cmd/link/internal/loadpe",
	"cmd/link/internal/loadxcoff",
	"cmd/link/internal/mips",
	"cmd/link/internal/mips64",
	"cmd/link/internal/objfile",
	"cmd/link/internal/ppc64",
	"cmd/link/internal/riscv64",
	"cmd/link/internal/s390x",
	"cmd/link/internal/sym",
	"cmd/link/internal/x86",
	"compress/flate",
	"compress/zlib",
	"cmd/link/internal/wasm",
	"container/heap",
	"debug/dwarf",
	"debug/elf",
	"debug/macho",
	"debug/pe",
	"internal/goversion",
	"internal/race",
	"internal/xcoff",
	"math/big",
	"math/bits",
	"sort",
}

// File prefixes that are ignored by go/build anyway, and cause
// problems with editor generated temporary files (#18931).
var ignorePrefixes = []string{
	".",
	"_",
}

// File suffixes that use build tags introduced since Go 1.4.
// These must not be copied into the bootstrap build directory.
var ignoreSuffixes = []string{
	"_arm64.s",
	"_arm64.go",
	"_wasm.s",
	"_wasm.go",
}

func bootstrapBuildTools() {
	goroot_bootstrap := os.Getenv("GOROOT_BOOTSTRAP")
	if goroot_bootstrap == "" {
		goroot_bootstrap = pathf("%s/go1.4", os.Getenv("HOME"))
	}
	xprintf("Building Go toolchain1 using %s.\n", goroot_bootstrap)

	mkzbootstrap(pathf("%s/src/cmd/internal/objabi/zbootstrap.go", goroot))

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
	writefile("module bootstrap\n", pathf("%s/%s", base, "go.mod"), 0)
	for _, dir := range bootstrapDirs {
		src := pathf("%s/src/%s", goroot, dir)
		dst := pathf("%s/%s", base, dir)
		xmkdirall(dst)
		if dir == "cmd/cgo" {
			// Write to src because we need the file both for bootstrap
			// and for later in the main build.
			mkzdefaultcc("", pathf("%s/zdefaultcc.go", src))
		}
	Dir:
		for _, name := range xreaddirfiles(src) {
			for _, pre := range ignorePrefixes {
				if strings.HasPrefix(name, pre) {
					continue Dir
				}
			}
			for _, suf := range ignoreSuffixes {
				if strings.HasSuffix(name, suf) {
					continue Dir
				}
			}
			srcFile := pathf("%s/%s", src, name)
			dstFile := pathf("%s/%s", dst, name)
			text := bootstrapRewriteFile(srcFile)
			writefile(text, dstFile, 0)
		}
	}

	// Set up environment for invoking Go 1.4 go command.
	// GOROOT points at Go 1.4 GOROOT,
	// GOPATH points at our bootstrap workspace,
	// GOBIN is empty, so that binaries are installed to GOPATH/bin,
	// and GOOS, GOHOSTOS, GOARCH, and GOHOSTOS are empty,
	// so that Go 1.4 builds whatever kind of binary it knows how to build.
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

	// Run Go 1.4 to build binaries. Use -gcflags=-l to disable inlining to
	// workaround bugs in Go 1.4's compiler. See discussion thread:
	// https://groups.google.com/d/msg/golang-dev/Ss7mCKsvk8w/Gsq7VYI0AwAJ
	// Use the math_big_pure_go build tag to disable the assembly in math/big
	// which may contain unsupported instructions.
	// Note that if we are using Go 1.10 or later as bootstrap, the -gcflags=-l
	// only applies to the final cmd/go binary, but that's OK: if this is Go 1.10
	// or later we don't need to disable inlining to work around bugs in the Go 1.4 compiler.
	cmd := []string{
		pathf("%s/bin/go", goroot_bootstrap),
		"install",
		"-gcflags=-l",
		"-tags=math_big_pure_go compiler_bootstrap",
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
// architecture that isn't for the current runtime.GOARCH.
//
// When unneeded is true archCaps is the rewrite base filename without
// the "rewrite" prefix or ".go" suffix: AMD64, 386, ARM, ARM64, etc.
func isUnneededSSARewriteFile(srcFile string) (archCaps string, unneeded bool) {
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
	if fileArch == os.Getenv("GOHOSTARCH") {
		return "", false
	}
	if fileArch == strings.TrimSuffix(runtime.GOARCH, "le") {
		return "", false
	}
	if fileArch == strings.TrimSuffix(os.Getenv("GOARCH"), "le") {
		return "", false
	}
	return archCaps, true
}

func bootstrapRewriteFile(srcFile string) string {
	// During bootstrap, generate dummy rewrite files for
	// irrelevant architectures. We only need to build a bootstrap
	// binary that works for the current runtime.GOARCH.
	// This saves 6+ seconds of bootstrap.
	if archCaps, ok := isUnneededSSARewriteFile(srcFile); ok {
		return fmt.Sprintf(`// Code generated by go tool dist; DO NOT EDIT.

package ssa

func rewriteValue%s(v *Value) bool { panic("unused during bootstrap") }
func rewriteBlock%s(b *Block) bool { panic("unused during bootstrap") }
`, archCaps, archCaps)
	}

	return bootstrapFixImports(srcFile)
}

func bootstrapFixImports(srcFile string) string {
	lines := strings.SplitAfter(readfile(srcFile), "\n")
	inBlock := false
	for i, line := range lines {
		if strings.HasPrefix(line, "import (") {
			inBlock = true
			continue
		}
		if inBlock && strings.HasPrefix(line, ")") {
			inBlock = false
			continue
		}
		if strings.HasPrefix(line, `import "`) || strings.HasPrefix(line, `import . "`) ||
			inBlock && (strings.HasPrefix(line, "\t\"") || strings.HasPrefix(line, "\t. \"")) {
			line = strings.Replace(line, `"cmd/`, `"bootstrap/cmd/`, -1)
			for _, dir := range bootstrapDirs {
				if strings.HasPrefix(dir, "cmd/") {
					continue
				}
				line = strings.Replace(line, `"`+dir+`"`, `"bootstrap/`+dir+`"`, -1)
			}
			lines[i] = line
		}
	}

	lines[0] = "// Code generated by go tool dist; DO NOT EDIT.\n// This is a bootstrap copy of " + srcFile + "\n\n//line " + srcFile + ":1\n" + lines[0]

	return strings.Join(lines, "")
}
