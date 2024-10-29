// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package syslist stores tables of OS and ARCH names that are
// (or at one point were) acceptable build targets.

package syslist

// Note that this file is read by internal/goarch/gengoarch.go and by
// internal/goos/gengoos.go. If you change this file, look at those
// files as well.

// KnownOS is the list of past, present, and future known GOOS values.
// Do not remove from this list, as it is used for filename matching.
// If you add an entry to this list, look at UnixOS, below.
var KnownOS = map[string]bool{
	"aix":       true,
	"android":   true,
	"darwin":    true,
	"dragonfly": true,
	"freebsd":   true,
	"hurd":      true,
	"illumos":   true,
	"ios":       true,
	"js":        true,
	"linux":     true,
	"nacl":      true,
	"netbsd":    true,
	"openbsd":   true,
	"plan9":     true,
	"solaris":   true,
	"wasip1":    true,
	"windows":   true,
	"zos":       true,
}

// UnixOS is the set of GOOS values matched by the "unix" build tag.
// This is not used for filename matching.
// This list also appears in cmd/dist/build.go.
var UnixOS = map[string]bool{
	"aix":       true,
	"android":   true,
	"darwin":    true,
	"dragonfly": true,
	"freebsd":   true,
	"hurd":      true,
	"illumos":   true,
	"ios":       true,
	"linux":     true,
	"netbsd":    true,
	"openbsd":   true,
	"solaris":   true,
}

// KnownArch is the list of past, present, and future known GOARCH values.
// Do not remove from this list, as it is used for filename matching.
var KnownArch = map[string]bool{
	"386":         true,
	"amd64":       true,
	"amd64p32":    true,
	"arm":         true,
	"armbe":       true,
	"arm64":       true,
	"arm64be":     true,
	"loong64":     true,
	"mips":        true,
	"mipsle":      true,
	"mips64":      true,
	"mips64le":    true,
	"mips64p32":   true,
	"mips64p32le": true,
	"ppc":         true,
	"ppc64":       true,
	"ppc64le":     true,
	"riscv":       true,
	"riscv64":     true,
	"s390":        true,
	"s390x":       true,
	"sparc":       true,
	"sparc64":     true,
	"wasm":        true,
}
