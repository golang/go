// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

// isErrReadOnlyFS reports whether err is a read-only filesystem error.
// Plan 9 reports errors as strings, so it always returns false.
func isErrReadOnlyFS(err error) bool {
	return false
}
