// rundir

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests that method calls through an interface always
// call the locally defined method localT.m independent
// at which embedding level it is and in which order
// embedding is done.

package ignored

