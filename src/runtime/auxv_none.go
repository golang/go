// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !linux && !darwin && !dragonfly && !freebsd && !netbsd && !solaris
// +build !linux,!darwin,!dragonfly,!freebsd,!netbsd,!solaris

package runtime

func sysargs(argc int32, argv **byte) {
}
