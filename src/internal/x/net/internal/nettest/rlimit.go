// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nettest

const defaultMaxOpenFiles = 256

// MaxOpenFiles returns the maximum number of open files for the
// caller's process.
func MaxOpenFiles() int { return maxOpenFiles() }
