// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le

package cpu

const cacheLineSize = 32

func initOptions() {
	options = []option{
		{Name: "msa", Feature: &MIPS64X.HasMSA},
	}
}
