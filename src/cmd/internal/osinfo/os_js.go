// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js

package osinfo

import (
	"fmt"
)

// Version returns the OS version name/number.
func Version() (string, error) {
	// Version detection on wasm varies depending on the underlying runtime
	// (browser, node, etc), nor is there a standard via something like
	// WASI (see https://go.dev/issue/31105). We could attempt multiple
	// combinations, but for now we leave this unimplemented for
	// simplicity.
	return "", fmt.Errorf("unimplemented")
}
