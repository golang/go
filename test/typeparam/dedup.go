// rundir -G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Note: this doesn't really test the deduplication of
// instantiations. It just provides an easy mechanism to build a
// binary that you can then check with objdump manually to make sure
// deduplication is happening. TODO: automate this somehow?

package ignored
