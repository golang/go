// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

const CacheLinePadSize = 64

func doinit() {
	options = []option{
		{Name: "aes", Feature: &ARM64.HasAES},
		{Name: "pmull", Feature: &ARM64.HasPMULL},
		{Name: "sha1", Feature: &ARM64.HasSHA1},
		{Name: "sha2", Feature: &ARM64.HasSHA2},
		{Name: "crc32", Feature: &ARM64.HasCRC32},
		{Name: "atomics", Feature: &ARM64.HasATOMICS},
		{Name: "cpuid", Feature: &ARM64.HasCPUID},
		{Name: "isNeoverseN1", Feature: &ARM64.IsNeoverseN1},
		{Name: "isZeus", Feature: &ARM64.IsZeus},
	}

	// arm64 uses different ways to detect CPU features at runtime depending on the operating system.
	osInit()
}

func getisar0() uint64

func getMIDR() uint64
