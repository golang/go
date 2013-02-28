// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

#pragma cgo_import_static x_cgo_setenv

void x_cgo_setenv(char**);
void (*_cgo_setenv)(char**) = x_cgo_setenv;
