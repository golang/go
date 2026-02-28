// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package osinfo

import (
	"errors"
	"fmt"
)

// Version returns the OS version name/number.
func Version() (string, error) {
	return "", fmt.Errorf("unable to determine OS version: %w", errors.ErrUnsupported)
}
