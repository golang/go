// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go test . -run=^TestGenerated$ -fix

package platform

// An OSArch is a pair of GOOS and GOARCH values indicating a platform.
type OSArch struct {
	GOOS, GOARCH string
}

func (p OSArch) String() string {
	return p.GOOS + "/" + p.GOARCH
}

// RaceDetectorSupported reports whether goos/goarch supports the race
// detector. There is a copy of this function in cmd/dist/test.go.
// Race detector only supports 48-bit VMA on arm64. But it will always
// return true for arm64, because we don't have VMA size information during
// the compile time.
func RaceDetectorSupported(goos, goarch string) bool {
	switch goos {
	case "linux":
		return goarch == "amd64" || goarch == "ppc64le" || goarch == "arm64" || goarch == "s390x"
	case "darwin":
		return goarch == "amd64" || goarch == "arm64"
	case "freebsd", "netbsd", "windows":
		return goarch == "amd64"
	default:
		return false
	}
}

// MSanSupported reports whether goos/goarch supports the memory
// sanitizer option.
func MSanSupported(goos, goarch string) bool {
	switch goos {
	case "linux":
		return goarch == "amd64" || goarch == "arm64" || goarch == "loong64"
	case "freebsd":
		return goarch == "amd64"
	default:
		return false
	}
}

// ASanSupported reports whether goos/goarch supports the address
// sanitizer option.
func ASanSupported(goos, goarch string) bool {
	switch goos {
	case "linux":
		return goarch == "arm64" || goarch == "amd64" || goarch == "loong64" || goarch == "riscv64" || goarch == "ppc64le"
	default:
		return false
	}
}

// FuzzSupported reports whether goos/goarch supports fuzzing
// ('go test -fuzz=.').
func FuzzSupported(goos, goarch string) bool {
	switch goos {
	case "darwin", "freebsd", "linux", "windows":
		return true
	default:
		return false
	}
}

// FuzzInstrumented reports whether fuzzing on goos/goarch uses coverage
// instrumentation. (FuzzInstrumented implies FuzzSupported.)
func FuzzInstrumented(goos, goarch string) bool {
	switch goarch {
	case "amd64", "arm64":
		// TODO(#14565): support more architectures.
		return FuzzSupported(goos, goarch)
	default:
		return false
	}
}

// MustLinkExternal reports whether goos/goarch requires external linking
// with or without cgo dependencies.
func MustLinkExternal(goos, goarch string, withCgo bool) bool {
	if withCgo {
		switch goarch {
		case "loong64", "mips", "mipsle", "mips64", "mips64le":
			// Internally linking cgo is incomplete on some architectures.
			// https://go.dev/issue/14449
			return true
		case "arm64":
			if goos == "windows" {
				// windows/arm64 internal linking is not implemented.
				return true
			}
		case "ppc64":
			// Big Endian PPC64 cgo internal linking is not implemented for aix or linux.
			// https://go.dev/issue/8912
			if goos == "aix" || goos == "linux" {
				return true
			}
		}

		switch goos {
		case "android":
			return true
		case "dragonfly":
			// It seems that on Dragonfly thread local storage is
			// set up by the dynamic linker, so internal cgo linking
			// doesn't work. Test case is "go test runtime/cgo".
			return true
		}
	}

	switch goos {
	case "android":
		if goarch != "arm64" {
			return true
		}
	case "ios":
		if goarch == "arm64" {
			return true
		}
	}
	return false
}

// BuildModeSupported reports whether goos/goarch supports the given build mode
// using the given compiler.
// There is a copy of this function in cmd/dist/test.go.
func BuildModeSupported(compiler, buildmode, goos, goarch string) bool {
	if compiler == "gccgo" {
		return true
	}

	if _, ok := distInfo[OSArch{goos, goarch}]; !ok {
		return false // platform unrecognized
	}

	platform := goos + "/" + goarch
	switch buildmode {
	case "archive":
		return true

	case "c-archive":
		switch goos {
		case "aix", "darwin", "ios", "windows":
			return true
		case "linux":
			switch goarch {
			case "386", "amd64", "arm", "armbe", "arm64", "arm64be", "loong64", "ppc64le", "riscv64", "s390x":
				// linux/ppc64 not supported because it does
				// not support external linking mode yet.
				return true
			default:
				// Other targets do not support -shared,
				// per ParseFlags in
				// cmd/compile/internal/base/flag.go.
				// For c-archive the Go tool passes -shared,
				// so that the result is suitable for inclusion
				// in a PIE or shared library.
				return false
			}
		case "freebsd":
			return goarch == "amd64"
		}
		return false

	case "c-shared":
		switch platform {
		case "linux/amd64", "linux/arm", "linux/arm64", "linux/loong64", "linux/386", "linux/ppc64le", "linux/riscv64", "linux/s390x",
			"android/amd64", "android/arm", "android/arm64", "android/386",
			"freebsd/amd64",
			"darwin/amd64", "darwin/arm64",
			"windows/amd64", "windows/386", "windows/arm64",
			"wasip1/wasm":
			return true
		}
		return false

	case "default":
		return true

	case "exe":
		return true

	case "pie":
		switch platform {
		case "linux/386", "linux/amd64", "linux/arm", "linux/arm64", "linux/loong64", "linux/ppc64le", "linux/riscv64", "linux/s390x",
			"android/amd64", "android/arm", "android/arm64", "android/386",
			"freebsd/amd64",
			"darwin/amd64", "darwin/arm64",
			"ios/amd64", "ios/arm64",
			"aix/ppc64",
			"openbsd/arm64",
			"windows/386", "windows/amd64", "windows/arm", "windows/arm64":
			return true
		}
		return false

	case "shared":
		switch platform {
		case "linux/386", "linux/amd64", "linux/arm", "linux/arm64", "linux/ppc64le", "linux/s390x":
			return true
		}
		return false

	case "plugin":
		switch platform {
		case "linux/amd64", "linux/arm", "linux/arm64", "linux/386", "linux/loong64", "linux/s390x", "linux/ppc64le",
			"android/amd64", "android/386",
			"darwin/amd64", "darwin/arm64",
			"freebsd/amd64":
			return true
		}
		return false

	default:
		return false
	}
}

func InternalLinkPIESupported(goos, goarch string) bool {
	switch goos + "/" + goarch {
	case "android/arm64",
		"darwin/amd64", "darwin/arm64",
		"linux/amd64", "linux/arm64", "linux/ppc64le",
		"windows/386", "windows/amd64", "windows/arm", "windows/arm64":
		return true
	}
	return false
}

// DefaultPIE reports whether goos/goarch produces a PIE binary when using the
// "default" buildmode. On Windows this is affected by -race,
// so force the caller to pass that in to centralize that choice.
func DefaultPIE(goos, goarch string, isRace bool) bool {
	switch goos {
	case "android", "ios":
		return true
	case "windows":
		if isRace {
			// PIE is not supported with -race on windows;
			// see https://go.dev/cl/416174.
			return false
		}
		return true
	case "darwin":
		return true
	}
	return false
}

// ExecutableHasDWARF reports whether the linked executable includes DWARF
// symbols on goos/goarch.
func ExecutableHasDWARF(goos, goarch string) bool {
	switch goos {
	case "plan9", "ios":
		return false
	}
	return true
}

// osArchInfo describes information about an OSArch extracted from cmd/dist and
// stored in the generated distInfo map.
type osArchInfo struct {
	CgoSupported bool
	FirstClass   bool
	Broken       bool
}

// CgoSupported reports whether goos/goarch supports cgo.
func CgoSupported(goos, goarch string) bool {
	return distInfo[OSArch{goos, goarch}].CgoSupported
}

// FirstClass reports whether goos/goarch is considered a “first class” port.
// (See https://go.dev/wiki/PortingPolicy#first-class-ports.)
func FirstClass(goos, goarch string) bool {
	return distInfo[OSArch{goos, goarch}].FirstClass
}

// Broken reports whether goos/goarch is considered a broken port.
// (See https://go.dev/wiki/PortingPolicy#broken-ports.)
func Broken(goos, goarch string) bool {
	return distInfo[OSArch{goos, goarch}].Broken
}
