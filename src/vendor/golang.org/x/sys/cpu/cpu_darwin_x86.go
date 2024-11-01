// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin && amd64 && gc

package cpu

// darwinSupportsAVX512 checks Darwin kernel for AVX512 support via sysctl
// call (see issue 43089). It also restricts AVX512 support for Darwin to
// kernel version 21.3.0 (MacOS 12.2.0) or later (see issue 49233).
//
// Background:
// Darwin implements a special mechanism to economize on thread state when
// AVX512 specific registers are not in use. This scheme minimizes state when
// preempting threads that haven't yet used any AVX512 instructions, but adds
// special requirements to check for AVX512 hardware support at runtime (e.g.
// via sysctl call or commpage inspection). See issue 43089 and link below for
// full background:
// https://github.com/apple-oss-distributions/xnu/blob/xnu-11215.1.10/osfmk/i386/fpu.c#L214-L240
//
// Additionally, all versions of the Darwin kernel from 19.6.0 through 21.2.0
// (corresponding to MacOS 10.15.6 - 12.1) have a bug that can cause corruption
// of the AVX512 mask registers (K0-K7) upon signal return. For this reason
// AVX512 is considered unsafe to use on Darwin for kernel versions prior to
// 21.3.0, where a fix has been confirmed. See issue 49233 for full background.
func darwinSupportsAVX512() bool {
	return darwinSysctlEnabled([]byte("hw.optional.avx512f\x00")) && darwinKernelVersionCheck(21, 3, 0)
}

// Ensure Darwin kernel version is at least major.minor.patch, avoiding dependencies
func darwinKernelVersionCheck(major, minor, patch int) bool {
	var release [256]byte
	err := darwinOSRelease(&release)
	if err != nil {
		return false
	}

	var mmp [3]int
	c := 0
Loop:
	for _, b := range release[:] {
		switch {
		case b >= '0' && b <= '9':
			mmp[c] = 10*mmp[c] + int(b-'0')
		case b == '.':
			c++
			if c > 2 {
				return false
			}
		case b == 0:
			break Loop
		default:
			return false
		}
	}
	if c != 2 {
		return false
	}
	return mmp[0] > major || mmp[0] == major && (mmp[1] > minor || mmp[1] == minor && mmp[2] >= patch)
}
