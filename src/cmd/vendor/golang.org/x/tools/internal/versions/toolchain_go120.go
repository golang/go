// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.20
// +build go1.20

package versions

func init() {
	if Compare(toolchain, Go1_20) < 0 {
		toolchain = Go1_20
	}
}
