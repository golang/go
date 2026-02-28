// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test preemption.

#include <stdlib.h>

#include "libgo8.h"

int main() {
	GoFunction8();

	// That should have exited the program.
	abort();
}
