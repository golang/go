// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dwarf

import (
	"internal/binarylite"
	"os"
	"reflect"
	"testing"
)

func TestDwarf5Ranges(t *testing.T) {
	rngLists, err := os.ReadFile("testdata/debug_rnglists")
	if err != nil {
		t.Fatalf("could not read test data: %v", err)
	}

	d := &Data{}
	d.order = binarylite.LittleEndian
	if err := d.AddSection(".debug_rnglists", rngLists); err != nil {
		t.Fatal(err)
	}
	u := &unit{
		asize: 8,
		vers:  5,
		is64:  true,
	}
	ret, err := d.dwarf5Ranges(u, nil, 0x5fbd, 0xc, [][2]uint64{})
	if err != nil {
		t.Fatalf("could not read rnglist: %v", err)
	}
	t.Logf("%#v", ret)

	tgt := [][2]uint64{{0x0000000000006712, 0x000000000000679f}, {0x00000000000067af}, {0x00000000000067b3}}

	if reflect.DeepEqual(ret, tgt) {
		t.Errorf("expected %#v got %#x", tgt, ret)
	}
}
