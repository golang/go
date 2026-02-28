// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noopt

package testenv

// OptimizationOff reports whether optimization is disabled.
func OptimizationOff() bool {
	return false
}
