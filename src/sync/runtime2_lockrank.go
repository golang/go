// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.staticlockranking
// +build goexperiment.staticlockranking

package sync

import "unsafe"

// Approximation of notifyList in runtime/sema.go. Size and alignment must
// agree.
type notifyList struct {
	wait   uint32
	notify uint32
	rank   int     // rank field of the mutex
	pad    int     // pad field of the mutex
	lock   uintptr // key field of the mutex

	head unsafe.Pointer
	tail unsafe.Pointer
}
