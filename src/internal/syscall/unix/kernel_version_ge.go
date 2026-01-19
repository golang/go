// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

// KernelVersionGE checks if the running kernel version
// is greater than or equal to the provided version.
func KernelVersionGE(x, y int) bool {
	xx, yy := KernelVersion()

	return xx > x || (xx == x && yy >= y)
}
