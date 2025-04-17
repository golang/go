// rundir

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 29610: Symbol import and initialization order caused function
// symbols to be recorded as non-function symbols.

// This uses rundir not because we actually want to run the final
// binary, but because we need to at least link it.

package ignored
