// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !freebsd && !linux && !solaris

package unix

func KernelVersion() (major int, minor int) {
	return 0, 0
}
