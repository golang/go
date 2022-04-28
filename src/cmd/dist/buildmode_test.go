//go:build go1.14
// +build go1.14

package main

import (
	"cmd/internal/sys"
	"runtime"
	"testing"
)

var osList = []string{"aix", "android", "darwin", "dragonfly", "freebsd",
	"hurd", "illumos", "ios", "js", "linux", "nacl", "netbsd",
	"openbsd", "plan9", "solaris", "windows", "zos"}

var archList = []string{
	"386", "amd64", "amd64p32", "arm", "armbe", "arm64", "arm64be",
	"loong64", "mips", "mipsle", "mips64", "mips64le", "mips64p32",
	"mips64p32le", "ppc", "ppc64", "ppc64le", "riscv", "riscv64",
	"s390", "s390x", "sparc", "sparc64", "wasm"}

var buildmodeList = []string{
	"archive", "c-archive", "c-shared", "default", "shared", "exe", "pie",
	"plugin"}

func TestBuildModeSupported(t *testing.T) {
	// Regression test for issue #43571. Checks that
	// (*cmd/dist.tester).supportedBuildmode and
	// cmd/internal/sys.BuildModeSupported are in sync.
	tester := &tester{} // none of the contents are actually used
	actualGoos := goos
	actualGoarch := goarch
	actualGohostos := gohostos
	gc := runtime.Compiler
	for _, os := range osList {
		for _, arch := range archList {
			for _, buildmode := range buildmodeList {
				goos = os
				goarch = arch
				gohostos = os
				testerResult := tester.supportedBuildmode(buildmode)
				sysResult := sys.BuildModeSupported(gc, buildmode, os, arch)
				if testerResult != sysResult {
					t.Errorf("Build mode mismatch for %s-%s %s:\n"+
						"tester.supportedBuildmode: %t\n"+
						"sys.BuildModeSupported: %t", os, arch, buildmode,
						testerResult, sysResult)
				}
			}
		}
	}
	goos = actualGoos
	goarch = actualGoarch
	gohostos = actualGohostos
}
