// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !linux,arm64

package cpu

const cacheLineSize = 64

func doinit() {}
