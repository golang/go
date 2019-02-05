// +build amd64,linux

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// Test that C.GoString uses IndexByte in safe manner.

/*
#include <sys/mman.h>

// Returns string with null byte at the last valid address
char* dangerousString1() {
	int pageSize = 4096;
	char *data = mmap(0, 2 * pageSize, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, 0, 0);
	mprotect(data + pageSize,pageSize,PROT_NONE);
	int start = pageSize - 123 - 1; // last 123 bytes of first page + 1 null byte
	int i = start;
	for (; i < pageSize; i++) {
	data[i] = 'x';
	}
	data[pageSize -1 ] = 0;
	return data+start;
}

char* dangerousString2() {
	int pageSize = 4096;
	char *data = mmap(0, 3 * pageSize, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, 0, 0);
	mprotect(data + 2 * pageSize,pageSize,PROT_NONE);
	int start = pageSize - 123 - 1; // last 123 bytes of first page + 1 null byte
	int i = start;
	for (; i < 2 * pageSize; i++) {
	data[i] = 'x';
	}
	data[2*pageSize -1 ] = 0;
	return data+start;
}
*/
import "C"

import (
	"testing"
)

func test24206(t *testing.T) {
	if l := len(C.GoString(C.dangerousString1())); l != 123 {
		t.Errorf("Incorrect string length - got %d, want 123", l)
	}
	if l := len(C.GoString(C.dangerousString2())); l != 4096+123 {
		t.Errorf("Incorrect string length - got %d, want %d", l, 4096+123)
	}
}
