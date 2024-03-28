// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rchan

import (
	"unsafe"
)

// A Select is a single case passed to rselect.
type Select struct {
	Dir SelectDir
	Typ unsafe.Pointer // channel type (not used here)
	Ch  unsafe.Pointer // channel
	Val unsafe.Pointer // ptr to data (SendDir) or ptr to receive buffer (RecvDir)
}

// These values must match ../reflect/value.go:/SelectDir.
type SelectDir int

const (
	_             SelectDir = iota
	SelectSend              // case Chan <- Send
	SelectRecv              // case <-Chan:
	SelectDefault           // default
)
