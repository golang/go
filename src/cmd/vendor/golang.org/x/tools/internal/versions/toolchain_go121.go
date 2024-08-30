// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.21
// +build go1.21

package versions

func init() {
	if Compare(toolchain, Go1_21) < 0 {
		toolchain = Go1_21
	}
}
