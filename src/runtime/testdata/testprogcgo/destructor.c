// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "_cgo_export.h"

static void callDestructorCallback() {
	GoDestructorCallback();
}

static void (*destructorFn)(void);

void registerDestructor() {
	destructorFn = callDestructorCallback;
}

__attribute__((destructor))
static void destructor() {
	if (destructorFn) {
		destructorFn();
	}
}
