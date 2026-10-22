// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.greenteagc && js && wasm

package runtime

import "internal/runtime/gc"

// logosSpanDeckOn is always false in wasm builds: the experiment's
// implementation is excluded there.
const logosSpanDeckOn = false

type logosSpanDeck struct{}

func (q *spanQueue) logosDeckInsert(objptr) bool { return false }
func logosSpanAscend([]objptr) {}
func (q *spanQueue) logosDeckPop() objptr { return 0 }
func logosSpanSort([]objptr) {}
func logosMarkPtrs(*gcWork, []uintptr) {}
func logosScanObject(uintptr, uintptr, uintptr, *gcWork) {}
func logosScanObjects(uintptr, uintptr, uint16, *gcWork, *gc.ObjMask) {}
