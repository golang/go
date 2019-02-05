// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

// maxGetRandomRead is the maximum number of bytes to ask for in one call to the
// getrandom() syscall. In linux at most 2^25-1 bytes will be returned per call.
// From the manpage
//
//	*  When reading from the urandom source, a maximum of 33554431 bytes
//	   is returned by a single call to getrandom() on systems where int
//	   has a size of 32 bits.
const maxGetRandomRead = (1 << 25) - 1
