// rundir

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 42401: linkname doesn't work correctly when a variable symbol
// is both imported (possibly through inlining) and linkname'd.

package ignored
