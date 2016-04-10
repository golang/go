// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
#include <stdint.h>

typedef struct {
	char x;
	long y;
} __attribute__((__packed__)) misaligned;

int
offset7560(void)
{
	return (uintptr_t)&((misaligned*)0)->y;
}
*/
import "C"

import (
	"reflect"
	"testing"
)

func test7560(t *testing.T) {
	// some mingw don't implement __packed__ correctly.
	if C.offset7560() != 1 {
		t.Skip("C compiler did not pack struct")
	}

	// C.misaligned should have x but then a padding field to get to the end of the struct.
	// There should not be a field named 'y'.
	var v C.misaligned
	rt := reflect.TypeOf(&v).Elem()
	if rt.NumField() != 2 || rt.Field(0).Name != "x" || rt.Field(1).Name != "_" {
		t.Errorf("unexpected fields in C.misaligned:\n")
		for i := 0; i < rt.NumField(); i++ {
			t.Logf("%+v\n", rt.Field(i))
		}
	}
}
