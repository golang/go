// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !linux
// +build ppc64 ppc64le

package cpu

const cacheLineSize = 128

func doinit() {}
