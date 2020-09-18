// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Export guts for testing.

package runtime

import "unsafe"

var NewOSProc0 = newosproc0
var Mincore = mincore
var Add = add

type EpollEvent epollevent

func Epollctl(epfd, op, fd int32, ev unsafe.Pointer) int32 {
	return epollctl(epfd, op, fd, (*epollevent)(ev))
}
