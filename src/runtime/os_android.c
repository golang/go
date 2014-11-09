// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"

// Export the runtime entry point symbol.
//
// Used by the app package to start the Go runtime after loading
// a shared library via JNI. See golang.org/x/mobile/app.

void _rt0_arm_linux1();
#pragma cgo_export_static _rt0_arm_linux1
#pragma cgo_export_dynamic _rt0_arm_linux1
