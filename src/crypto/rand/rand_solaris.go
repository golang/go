// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

// maxGetRandomRead is the maximum number of bytes to ask for in one call to the
// getrandom() syscall. Across all the Solaris platforms, 256 bytes is the
// lowest number of bytes returned atomically per call.
const maxGetRandomRead = 1 << 8
