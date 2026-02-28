// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

// Export for testing.

var Atime = atime
var ErrWriteAtInAppendMode = errWriteAtInAppendMode
var ErrPatternHasSeparator = errPatternHasSeparator

func init() {
	checkWrapErr = true
}

var ExportReadFileContents = readFileContents

// cleanuper stands in for *testing.T, since we can't import testing in os.
type cleanuper interface {
	Cleanup(func())
}

func SetStatHook(t cleanuper, f func(f *File, name string) (FileInfo, error)) {
	oldstathook := stathook
	t.Cleanup(func() {
		stathook = oldstathook
	})
	stathook = f
}
