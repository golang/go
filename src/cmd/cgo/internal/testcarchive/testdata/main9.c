// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "libgo9.h"

void use(int *x) { (*x)++; }

void callGoFWithDeepStack(int p) {
	int x[10000];

	use(&x[0]);
	use(&x[9999]);

	GoF(p);

	use(&x[0]);
	use(&x[9999]);
}

void callGoWithVariousStack(int p) {
	GoF(0);                  // call GoF without using much stack
	callGoFWithDeepStack(p); // call GoF with a deep stack
	GoF(0);                  // again on a shallow stack
}

int main() {
	callGoWithVariousStack(0);

	callGoWithVariousStackAndGoFrame(0); // normal execution
	callGoWithVariousStackAndGoFrame(1); // panic and recover
}
