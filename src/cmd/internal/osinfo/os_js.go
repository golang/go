// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js

package osinfo

import (
	"fmt"
)

// Version returns the OS version name/number.
func Version() (string, error) {
	// TODO(prattmic): Does wasm have any version/runtime detection
	// functionality?
	return "", fmt.Errorf("unimplemented")
}
