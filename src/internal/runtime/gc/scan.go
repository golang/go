// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "internal/goarch"

// ObjMask is a bitmap where each bit corresponds to an object in a span.
//
// It is sized to accommodate all size classes.
type ObjMask [MaxObjsPerSpan / (goarch.PtrSize * 8)]uintptr

// PtrMask is a bitmap where each bit represents a pointer-word in a single runtime page.
type PtrMask [PageSize / goarch.PtrSize / (goarch.PtrSize * 8)]uintptr
