// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"
)

func usage() {
	xprintf(`usage: go tool dist [command]
Commands are:

banner         print installation banner
bootstrap      rebuild everything
clean          deletes all built files
env [-p]       print environment (-p: include $PATH)
install [dir]  install individual directory
list [-json]   list all supported platforms
test [-h]      run Go test(s)
version        print Go version

All commands take -v flags to emit extra information.
`)
	xexit(2)
}

// commands records the available commands.
var commands = map[string]func(){
	"banner":    cmdbanner,
	"bootstrap": cmdbootstrap,
	"clean":     cmdclean,
	"env":       cmdenv,
	"install":   cmdinstall,
	"list":      cmdlist,
	"test":      cmdtest,
	"version":   cmdversion,
}

// main takes care of OS-specific startup and dispatches to xmain.
func main() {
	os.Setenv("TERM", "dumb") // disable escape codes in clang errors

	// provide -check-armv6k first, before checking for $GOROOT so that
	// it is possible to run this check without having $GOROOT available.
	if len(os.Args) > 1 && os.Args[1] == "-check-armv6k" {
		useARMv6K() // might fail with SIGILL
		println("ARMv6K supported.")
		os.Exit(0)
	}

	gohostos = runtime.GOOS
	switch gohostos {
	case "aix":
		// uname -m doesn't work under AIX
		gohostarch = "ppc64"
	case "darwin":
		// macOS 10.9 and later require clang
		defaultclang = true
	case "freebsd":
		// Since FreeBSD 10 gcc is no longer part of the base system.
		defaultclang = true
	case "openbsd":
		// OpenBSD ships with GCC 4.2, which is now quite old.
		defaultclang = true
	case "plan9":
		gohostarch = os.Getenv("objtype")
		if gohostarch == "" {
			fatalf("$objtype is unset")
		}
	case "solaris", "illumos":
		// Solaris and illumos systems have multi-arch userlands, and
		// "uname -m" reports the machine hardware name; e.g.,
		// "i86pc" on both 32- and 64-bit x86 systems.  Check for the
		// native (widest) instruction set on the running kernel:
		out := run("", CheckExit, "isainfo", "-n")
		if strings.Contains(out, "amd64") {
			gohostarch = "amd64"
		}
		if strings.Contains(out, "i386") {
			gohostarch = "386"
		}
	case "windows":
		exe = ".exe"
	}

	sysinit()

	if gohostarch == "" {
		// Default Unix system.
		out := run("", CheckExit, "uname", "-m")
		outAll := run("", CheckExit, "uname", "-a")
		switch {
		case strings.Contains(outAll, "RELEASE_ARM64"):
			// MacOS prints
			// Darwin p1.local 21.1.0 Darwin Kernel Version 21.1.0: Wed Oct 13 17:33:01 PDT 2021; root:xnu-8019.41.5~1/RELEASE_ARM64_T6000 x86_64
			// on ARM64 laptops when there is an x86 parent in the
			// process tree. Look for the RELEASE_ARM64 to avoid being
			// confused into building an x86 toolchain.
			gohostarch = "arm64"
		case strings.Contains(out, "x86_64"), strings.Contains(out, "amd64"):
			gohostarch = "amd64"
		case strings.Contains(out, "86"):
			gohostarch = "386"
			if gohostos == "darwin" {
				// Even on 64-bit platform, some versions of macOS uname -m prints i386.
				// We don't support any of the OS X versions that run on 32-bit-only hardware anymore.
				gohostarch = "amd64"
			}
		case strings.Contains(out, "aarch64"), strings.Contains(out, "arm64"):
			gohostarch = "arm64"
		case strings.Contains(out, "arm"):
			gohostarch = "arm"
			if gohostos == "netbsd" && strings.Contains(run("", CheckExit, "uname", "-p"), "aarch64") {
				gohostarch = "arm64"
			}
		case strings.Contains(out, "ppc64le"):
			gohostarch = "ppc64le"
		case strings.Contains(out, "ppc64"):
			gohostarch = "ppc64"
		case strings.Contains(out, "mips64"):
			gohostarch = "mips64"
			if elfIsLittleEndian(os.Args[0]) {
				gohostarch = "mips64le"
			}
		case strings.Contains(out, "mips"):
			gohostarch = "mips"
			if elfIsLittleEndian(os.Args[0]) {
				gohostarch = "mipsle"
			}
		case strings.Contains(out, "riscv64"):
			gohostarch = "riscv64"
		case strings.Contains(out, "s390x"):
			gohostarch = "s390x"
		case gohostos == "darwin", gohostos == "ios":
			if strings.Contains(run("", CheckExit, "uname", "-v"), "RELEASE_ARM64_") {
				gohostarch = "arm64"
			}
		case gohostos == "openbsd":
			if strings.Contains(run("", CheckExit, "uname", "-p"), "mips64") {
				gohostarch = "mips64"
			}
		default:
			fatalf("unknown architecture: %s", out)
		}
	}

	if gohostarch == "arm" || gohostarch == "mips64" || gohostarch == "mips64le" {
		maxbg = min(maxbg, runtime.NumCPU())
	}
	bginit()

	if len(os.Args) > 1 && os.Args[1] == "-check-goarm" {
		useVFPv1() // might fail with SIGILL
		println("VFPv1 OK.")
		useVFPv3() // might fail with SIGILL
		println("VFPv3 OK.")
		os.Exit(0)
	}

	xinit()
	xmain()
	xexit(0)
}

// The OS-specific main calls into the portable code here.
func xmain() {
	if len(os.Args) < 2 {
		usage()
	}
	cmd := os.Args[1]
	os.Args = os.Args[1:] // for flag parsing during cmd
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: go tool dist %s [options]\n", cmd)
		flag.PrintDefaults()
		os.Exit(2)
	}
	if f, ok := commands[cmd]; ok {
		f()
	} else {
		xprintf("unknown command %s\n", cmd)
		usage()
	}
}
