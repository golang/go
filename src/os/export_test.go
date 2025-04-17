// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

// Export for testing.

var Atime = atime
var LstatP = &lstat
var ErrWriteAtInAppendMode = errWriteAtInAppendMode
var ErrPatternHasSeparator = errPatternHasSeparator

func init() {
	checkWrapErr = true
}

var ExportReadFileContents = readFileContents
