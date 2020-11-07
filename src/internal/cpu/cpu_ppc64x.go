// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

package cpu

const CacheLinePadSize = 128

func doinit() {
	options = []option{
		{Name: "darn", Feature: &PPC64.HasDARN},
		{Name: "scv", Feature: &PPC64.HasSCV},
		{Name: "power9", Feature: &PPC64.IsPOWER9},
	}

	osinit()
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}
