// rundir

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 19764: test that the linker's trampoline insertion
// pass is happy with direct calls to interface wrappers that
// may be defined in multiple packages.
package ignore
