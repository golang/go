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
	"os"
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
	"cmd/compile",
	"cmd/compile/internal/amd64",
	"cmd/compile/internal/arm",
	"cmd/compile/internal/arm64",
	"cmd/compile/internal/gc",
	"cmd/compile/internal/mips",
	"cmd/compile/internal/mips64",
	"cmd/compile/internal/ppc64",
	"cmd/compile/internal/s390x",
	"cmd/compile/internal/ssa",
	"cmd/compile/internal/syntax",
	"cmd/compile/internal/x86",
	"cmd/internal/bio",
	"cmd/internal/gcprog",
	"cmd/internal/dwarf",
	"cmd/internal/obj",
	"cmd/internal/obj/arm",
	"cmd/internal/obj/arm64",
	"cmd/internal/obj/mips",
	"cmd/internal/obj/ppc64",
	"cmd/internal/obj/s390x",
	"cmd/internal/obj/x86",
	"cmd/internal/sys",
	"cmd/link",
	"cmd/link/internal/amd64",
	"cmd/link/internal/arm",
	"cmd/link/internal/arm64",
	"cmd/link/internal/ld",
	"cmd/link/internal/mips",
	"cmd/link/internal/mips64",
	"cmd/link/internal/ppc64",
	"cmd/link/internal/s390x",
	"cmd/link/internal/x86",
	"debug/pe",
	"math/big",
}

// File suffixes that use build tags introduced since Go 1.4.
// These must not be copied into the bootstrap build directory.
var ignoreSuffixes = []string{
	"_arm64.s",
	"_arm64.go",
}

func bootstrapBuildTools() {
	goroot_bootstrap := os.Getenv("GOROOT_BOOTSTRAP")
	if goroot_bootstrap == "" {
		goroot_bootstrap = pathf("%s/go1.4", os.Getenv("HOME"))
	}
	xprintf("##### Building Go toolchain using %s.\n", goroot_bootstrap)

	mkzbootstrap(pathf("%s/src/cmd/internal/obj/zbootstrap.go", goroot))

	// Use $GOROOT/pkg/bootstrap as the bootstrap workspace root.
	// We use a subdirectory of $GOROOT/pkg because that's the
	// space within $GOROOT where we store all generated objects.
	// We could use a temporary directory outside $GOROOT instead,
	// but it is easier to debug on failure if the files are in a known location.
	workspace := pathf("%s/pkg/bootstrap", goroot)
	xremoveall(workspace)
	base := pathf("%s/src/bootstrap", workspace)
	xmkdirall(base)

	// Copy source code into $GOROOT/pkg/bootstrap and rewrite import paths.
	for _, dir := range bootstrapDirs {
		src := pathf("%s/src/%s", goroot, dir)
		dst := pathf("%s/%s", base, dir)
		xmkdirall(dst)
	Dir:
		for _, name := range xreaddirfiles(src) {
			for _, suf := range ignoreSuffixes {
				if strings.HasSuffix(name, suf) {
					continue Dir
				}
			}
			srcFile := pathf("%s/%s", src, name)
			text := readfile(srcFile)
			text = bootstrapFixImports(text, srcFile)
			writefile(text, pathf("%s/%s", dst, name), 0)
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
	run(workspace, ShowOutput|CheckExit, pathf("%s/bin/go", goroot_bootstrap), "install", "-gcflags=-l", "-tags=math_big_pure_go", "-v", "bootstrap/cmd/...")

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

	xprintf("\n")
}

func bootstrapFixImports(text, srcFile string) string {
	lines := strings.SplitAfter(text, "\n")
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

	lines[0] = "// Do not edit. Bootstrap copy of " + srcFile + "\n\n//line " + srcFile + ":1\n" + lines[0]

	return strings.Join(lines, "")
}
