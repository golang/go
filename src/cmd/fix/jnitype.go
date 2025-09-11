// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	register(jniFix)
}

var jniFix = fix{
	name:     "jni",
	date:     "2017-12-04",
	f:        noop,
	desc:     `Fixes initializers of JNI's jobject and subtypes (removed)`,
	disabled: false,
}
