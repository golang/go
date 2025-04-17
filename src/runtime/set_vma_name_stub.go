// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !linux

package runtime

import "unsafe"

// setVMAName isn’t implemented
func setVMAName(start unsafe.Pointer, len uintptr, name string) {}
