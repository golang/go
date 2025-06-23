// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm64 && linux && !android

package cpu

func osInit() {
	hwcapInit("linux")
}
