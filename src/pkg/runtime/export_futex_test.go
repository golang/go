// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux freebsd

package runtime

func futexsleep(addr *uint32, val uint32, ns int64)
func futexwakeup(addr *uint32, val uint32)

var Futexsleep = futexsleep
var Futexwakeup = futexwakeup
