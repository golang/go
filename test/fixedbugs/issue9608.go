// rundir

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 9608: dead code elimination in switch statements.

// This has to be done as a package rather than as a file,
// because run.go runs files with 'go run', which passes the
// -complete flag to compiler, causing it to complain about
// the intentionally unimplemented function fail.

package ignored
