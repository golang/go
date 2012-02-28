// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build netbsd

package net

func setIPv4MulticastInterface(fd *netFD, ifi *Interface) error {
	panic("unimplemented")
}

func setIPv4MulticastLoopback(fd *netFD, v bool) error {
	panic("unimplemented")
}
