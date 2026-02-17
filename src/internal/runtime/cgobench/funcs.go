// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

package cgobench

/*
static void empty() {
}

void go_empty_callback();

static void callback() {
	go_empty_callback();
}

*/
import "C"

func EmptyC() {
	C.empty()
}

func CallbackC() {
	C.callback()
}

//export go_empty_callback
func go_empty_callback() {
}

//go:noinline
func Empty() {
}
