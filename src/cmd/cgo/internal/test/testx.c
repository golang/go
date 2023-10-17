// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "_cgo_export.h"

void lockOSThreadC(void) {
	lockOSThreadCallback();
}

void issue7978c(uint32_t *sync) {
	while(__atomic_load_n(sync, __ATOMIC_SEQ_CST) != 0)
		;
	__atomic_add_fetch(sync, 1, __ATOMIC_SEQ_CST);
	while(__atomic_load_n(sync, __ATOMIC_SEQ_CST) != 2)
		;
	issue7978cb();
	__atomic_add_fetch(sync, 1, __ATOMIC_SEQ_CST);
	while(__atomic_load_n(sync, __ATOMIC_SEQ_CST) != 6)
		;
}

void f7665(void) {
}
