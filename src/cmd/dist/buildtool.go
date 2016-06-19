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
// All directories in this list are relative to and must be below $GOROOT/src/cmd.
// The list is assumed to have two kinds of entries: names without slashes,
// which are commands, and entries beginning with internal/, which are
// packages supporting the commands.
var bootstrapDirs = []string{
	"asm",
	"asm/internal/arch",
	"asm/internal/asm",
	"asm/internal/flags",
	"asm/internal/lex",
	"compile",
	"compile/internal/amd64",
	"compile/internal/arm",
	"compile/internal/arm64",
	"compile/internal/big",
	"compile/internal/gc",
	"compile/internal/mips64",
	"compile/internal/ppc64",
	"compile/internal/ssa",
	"compile/internal/x86",
	"compile/internal/s390x",
	"internal/bio",
	"internal/gcprog",
	"internal/obj",
	"internal/obj/arm",
	"internal/obj/arm64",
	"internal/obj/mips",
	"internal/obj/ppc64",
	"internal/obj/s390x",
	"internal/obj/x86",
	"internal/sys",
	"link",
	"link/internal/amd64",
	"link/internal/arm",
	"link/internal/arm64",
	"link/internal/ld",
	"link/internal/mips64",
	"link/internal/ppc64",
	"link/internal/s390x",
	"link/internal/x86",
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
		src := pathf("%s/src/cmd/%s", goroot, dir)
		dst := pathf("%s/%s", base, dir)
		xmkdirall(dst)
		for _, name := range xreaddirfiles(src) {
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
	run(workspace, ShowOutput|CheckExit, pathf("%s/bin/go", goroot_bootstrap), "install", "-gcflags=-l", "-v", "bootstrap/...")

	// Copy binaries into tool binary directory.
	for _, name := range bootstrapDirs {
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
			lines[i] = strings.Replace(line, `"cmd/`, `"bootstrap/`, -1)
		}
	}

	lines[0] = "// Do not edit. Bootstrap copy of " + srcFile + "\n\n//line " + srcFile + ":1\n" + lines[0]

	return strings.Join(lines, "")
}
