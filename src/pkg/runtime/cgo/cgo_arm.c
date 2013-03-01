// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma cgo_import_static x_cgo_load_gm
extern void x_cgo_load_gm(void);
void (*_cgo_load_gm)(void) = x_cgo_load_gm;

#pragma cgo_import_static x_cgo_save_gm
extern void x_cgo_save_gm(void);
void (*_cgo_save_gm)(void) = x_cgo_save_gm;

