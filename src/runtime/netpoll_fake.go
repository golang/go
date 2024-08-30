// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fake network poller for js/wasm.
// Should never be used, because js/wasm network connections do not honor "SetNonblock".

//go:build js && wasm

package runtime

func netpollinit() {
}

func netpollIsPollDescriptor(fd uintptr) bool {
	return false
}

func netpollopen(fd uintptr, pd *pollDesc) int32 {
	return 0
}

func netpollclose(fd uintptr) int32 {
	return 0
}

func netpollarm(pd *pollDesc, mode int) {
}

func netpollBreak() {
}

func netpoll(delay int64) (gList, int32) {
	return gList{}, 0
}
