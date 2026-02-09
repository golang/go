// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Export guts for testing on windows.

package poll

func SkipsCompletionPortOnSuccess(fd *FD) bool {
	return !fd.waitOnSuccess
}
