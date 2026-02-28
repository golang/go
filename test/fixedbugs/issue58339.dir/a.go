// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

func Assert(msgAndArgs ...any) {
}

func Run() int {
	Assert("%v")
	return 0
}

func Run2() int {
	return Run()
}
