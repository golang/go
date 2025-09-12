// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	register(eglFixDisplay)
	register(eglFixConfig)
}

var eglFixDisplay = fix{
	name:     "egl",
	date:     "2018-12-15",
	f:        noop,
	desc:     `Fixes initializers of EGLDisplay (removed)`,
	disabled: false,
}

var eglFixConfig = fix{
	name:     "eglconf",
	date:     "2020-05-30",
	f:        noop,
	desc:     `Fixes initializers of EGLConfig (removed)`,
	disabled: false,
}
