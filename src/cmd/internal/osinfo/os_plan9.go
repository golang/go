// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build plan9

package osinfo

import (
	"os"
)

// Version returns the OS version name/number.
func Version() (string, error) {
	b, err := os.ReadFile("/dev/osversion")
	if err != nil {
		return "", err
	}

	return string(b), nil
}
