// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package cgo contains runtime support for code generated
by the cgo tool.  See the documentation for the cgo command
for details on using cgo.
*/
package cgo

/*

#cgo darwin,!arm64 LDFLAGS: -lpthread
#cgo darwin,arm64 LDFLAGS: -framework CoreFoundation
#cgo dragonfly LDFLAGS: -lpthread
#cgo freebsd LDFLAGS: -lpthread
#cgo android LDFLAGS: -llog
#cgo !android,linux LDFLAGS: -lpthread
#cgo netbsd LDFLAGS: -lpthread
#cgo openbsd LDFLAGS: -lpthread
#cgo aix LDFLAGS: -Wl,-berok
#cgo solaris LDFLAGS: -lxnet
#cgo solaris LDFLAGS: -lsocket

// Use -fno-stack-protector to avoid problems locating the
// proper support functions. See issues #52919, #54313, #58385.
// Use -Wdeclaration-after-statement because some CI builds use it.
#cgo CFLAGS: -Wall -Werror -fno-stack-protector -Wdeclaration-after-statement

#cgo solaris CPPFLAGS: -D_POSIX_PTHREAD_SEMANTICS

*/
import "C"

import "runtime/internal/sys"

// Incomplete is used specifically for the semantics of incomplete C types.
type Incomplete struct {
	_ sys.NotInHeap
}
