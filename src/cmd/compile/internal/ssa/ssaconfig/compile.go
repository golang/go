// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssaconfig

// Debug output
var IntrinsicsDebug int

var IntrinsicsDisable bool

var BuildDebug int

var BuildTest int

var BuildStats int

var BuildDump map[string]bool = make(map[string]bool) // names of functions to dump after initial build of ssa

var GenssaDump map[string]bool = make(map[string]bool) // names of functions to dump after ssa has been converted to asm
