// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "libgo9.h"

void use(int *x) { (*x)++; }

void callGoFWithDeepStack() {
	int x[10000];

	use(&x[0]);
	use(&x[9999]);

	GoF();

	use(&x[0]);
	use(&x[9999]);
}

int main() {
	GoF();                  // call GoF without using much stack
	callGoFWithDeepStack(); // call GoF with a deep stack
}
