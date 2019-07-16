// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build lldb
// +build darwin
// +build arm arm64

// Used by gcc_signal_darwin_armx.c when doing the test build during cgo.
// We hope that for real binaries the definition provided by Go will take precedence
// and the linker will drop this .o file altogether, which is why this definition
// is all by itself in its own file.
void __attribute__((weak)) xx_cgo_panicmem(void) {}
