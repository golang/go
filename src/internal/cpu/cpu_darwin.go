// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin && !ios

package cpu

import _ "unsafe" // for linkname

// Pushed from runtime.
//
//go:noescape
func sysctlbynameInt32(name []byte) (int32, int32)

// Pushed from runtime.
//
//go:noescape
func sysctlbynameBytes(name, out []byte) int32

// sysctlEnabled should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/bytedance/gopkg
//   - github.com/songzhibin97/gkit
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname sysctlEnabled
func sysctlEnabled(name []byte) bool {
	ret, value := sysctlbynameInt32(name)
	if ret < 0 {
		return false
	}
	return value > 0
}

// darwinKernelVersionCheck reports if Darwin kernel version is at
// least major.minor.patch.
//
// Code borrowed from x/sys/cpu.
func darwinKernelVersionCheck(major, minor, patch int) bool {
	var release [256]byte
	ret := sysctlbynameBytes([]byte("kern.osrelease\x00"), release[:])
	if ret < 0 {
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
