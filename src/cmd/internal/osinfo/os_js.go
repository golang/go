// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js

package osinfo

import (
	"fmt"
	"syscall/js"
)

// Version returns the OS version name/number.
func Version() (string, error) {
	// Version detection on Wasm varies depending on the underlying runtime
	// (browser, node, etc), nor is there a standard via something like
	// WASI (see https://go.dev/issue/31105). For now, attempt a few simple
	// combinations for the convenience of reading logs at build.golang.org
	// and local development. It's not a goal to recognize all environments.
	if v, ok := node(); ok {
		return "Node.js " + v, nil
	}
	return "", fmt.Errorf("unrecognized environment")
}

func node() (version string, ok bool) {
	// Try the https://nodejs.org/api/process.html#processversion API.
	p := js.Global().Get("process")
	if p.IsUndefined() {
		return "", false
	}
	v := p.Get("version")
	if v.IsUndefined() {
		return "", false
	}
	return v.String(), true
}
