// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm

package filepathlite

import "syscall/js"

func IsPathSeparator(c uint8) bool {
	return c == '\\' || c == '/'
}

var jsPath = js.Global().Get("path")

var (
	Separator = jsPath.Get("sep").String()[0]
)

func isLocal(path string) bool {
	return !jsPath.Call("isAbsolute", path).Bool()
}

func localize(path string) (string, error) {
	return jsPath.Call("normalize", path).String(), nil
}

func IsAbs(path string) bool {
	return jsPath.Call("isAbsolute", path).Bool()
}

func volumeNameLen(path string) int {
	length := jsPath.Call("parse", path).Get("root").Get("length").Int()
	return max(0, length-1)
}
