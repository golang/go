// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (386 || amd64 || arm || arm64) && !purego

package checktest

import "unsafe"

func PtrStaticData() *uint32
func PtrStaticText() unsafe.Pointer
