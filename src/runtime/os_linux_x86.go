// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build 386 amd64

package runtime

//go:noescape
func uname(utsname *new_utsname) int

func mlock(addr, len uintptr) int

func osArchInit() {
	// Linux 5.2 introduced a bug that can corrupt vector
	// registers on return from a signal if the signal stack isn't
	// faulted in:
	// https://bugzilla.kernel.org/show_bug.cgi?id=205663
	//
	// It was fixed in 5.3.15, 5.4.2, and all 5.5 and later
	// kernels.
	//
	// If we're on an affected kernel, work around this issue by
	// mlocking the top page of every signal stack. This doesn't
	// help for signal stacks created in C, but there's not much
	// we can do about that.
	//
	// TODO(austin): Remove this in Go 1.15, at which point it
	// will be unlikely to encounter any of the affected kernels
	// in the wild.

	var uts new_utsname
	if uname(&uts) < 0 {
		throw("uname failed")
	}
	// Check for null terminator to ensure gostringnocopy doesn't
	// walk off the end of the release string.
	found := false
	for _, b := range uts.release {
		if b == 0 {
			found = true
			break
		}
	}
	if !found {
		return
	}
	rel := gostringnocopy(&uts.release[0])

	major, minor, patch, ok := parseRelease(rel)
	if !ok {
		return
	}

	if major == 5 && (minor == 2 || minor == 3 && patch < 15 || minor == 4 && patch < 2) {
		gsignalInitQuirk = mlockGsignal
		if m0.gsignal != nil {
			throw("gsignal quirk too late")
		}
	}
}

func mlockGsignal(gsignal *g) {
	if err := mlock(gsignal.stack.hi-physPageSize, physPageSize); err < 0 {
		printlock()
		println("runtime: mlock of signal stack failed:", -err)
		if err == -_ENOMEM {
			println("runtime: increase the mlock limit (ulimit -l) or")
		}
		println("runtime: update your kernel to 5.3.15+, 5.4.2+, or 5.5+")
		throw("mlock failed")
	}
}
