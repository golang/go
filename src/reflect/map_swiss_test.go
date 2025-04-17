// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.swissmap

package reflect_test

import (
	"reflect"
	"testing"
)

func testGCBitsMap(t *testing.T) {
	// Unlike old maps, we don't manually construct GC data for swiss maps,
	// instead using the public reflect API in groupAndSlotOf.
}

// See also runtime_test.TestGroupSizeZero.
func TestGroupSizeZero(t *testing.T) {
	st := reflect.TypeFor[struct{}]()
	grp := reflect.MapGroupOf(st, st)

	// internal/runtime/maps when create pointers to slots, even if slots
	// are size 0. We should have reserved an extra word to ensure that
	// pointers to the zero-size type at the end of group are valid.
	if grp.Size() <= 8 {
		t.Errorf("Group size got %d want >8", grp.Size())
	}
}
