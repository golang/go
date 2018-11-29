// rundir

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The compiler was using an incomplete symbol name for reflect name data,
// permitting an invalid merge in the linker, producing an incorrect
// exported flag bit.

package ignored
