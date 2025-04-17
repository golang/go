// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.swissmap

package maps

import (
	"internal/abi"
	"unsafe"
)

// For testing, we don't ever need key errors.
func mapKeyError(typ *abi.SwissMapType, p unsafe.Pointer) error {
	return nil
}
