// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd

package runtime

var NonblockingPipe = nonblockingPipe
var Pipe = pipe
var SetNonblock = setNonblock
var Closeonexec = closeonexec
