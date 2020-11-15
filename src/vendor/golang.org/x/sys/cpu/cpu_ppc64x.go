// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

package cpu

const cacheLineSize = 128

func initOptions() {
	options = []option{
		{Name: "darn", Feature: &PPC64.HasDARN},
		{Name: "scv", Feature: &PPC64.HasSCV},
	}
}
