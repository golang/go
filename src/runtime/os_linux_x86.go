// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build 386 amd64

package runtime

import "runtime/internal/atomic"

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
		throwReportQuirk = throwBadKernel
	}
}

func mlockGsignal(gsignal *g) {
	if atomic.Load(&touchStackBeforeSignal) != 0 {
		// mlock has already failed, don't try again.
		return
	}

	// This mlock call may fail, but we don't report the failure.
	// Instead, if something goes badly wrong, we rely on prepareSignalM
	// and throwBadKernel to do further mitigation and to report a problem
	// to the user if mitigation fails. This is because many
	// systems have a limit on the total mlock size, and many kernels
	// that appear to have bad versions are actually patched to avoid the
	// bug described above. We want Go 1.14 to run on those systems.
	// See #37436.
	if errno := mlock(gsignal.stack.hi-physPageSize, physPageSize); errno < 0 {
		atomic.Store(&touchStackBeforeSignal, uint32(-errno))
	}
}

// throwBadKernel is called, via throwReportQuirk, by throw.
func throwBadKernel() {
	if errno := atomic.Load(&touchStackBeforeSignal); errno != 0 {
		println("runtime: note: your Linux kernel may be buggy")
		println("runtime: note: see https://golang.org/wiki/LinuxKernelSignalVectorBug")
		println("runtime: note: mlock workaround for kernel bug failed with errno", errno)
	}
}
