// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dwarf

import (
	"reflect"
	"testing"
)

func TestMergeRangesNested(t *testing.T) {
	// An inline anchor inside an existing range must not truncate its end.
	outer, inner := []Range{{10, 30}}, []Range{{15, 20}}
	if got := MergeRanges(outer, inner); !reflect.DeepEqual(got, outer) {
		t.Fatalf("MergeRanges(%v, %v) = %v, want %v", outer, inner, got, outer)
	}
}

func TestSevenBitEnc128(t *testing.T) {
	t.Run("unsigned", func(t *testing.T) {
		for v := int64(-255); v < 255; v++ {
			s := sevenBitU(v)
			if s == nil {
				continue
			}
			b := AppendUleb128(nil, uint64(v))
			if !reflect.DeepEqual(b, s) {
				t.Errorf("sevenBitU(%d) = %v but AppendUleb128(%d) = %v", v, s, v, b)
			}
		}
	})

	t.Run("signed", func(t *testing.T) {
		for v := int64(-255); v < 255; v++ {
			s := sevenBitS(v)
			if s == nil {
				continue
			}
			b := AppendSleb128(nil, v)
			if !reflect.DeepEqual(b, s) {
				t.Errorf("sevenBitS(%d) = %v but AppendSleb128(%d) = %v", v, s, v, b)
			}
		}
	})
}
