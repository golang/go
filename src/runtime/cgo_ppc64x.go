// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

package runtime

// crosscall_ppc64 calls into the runtime to set up the registers the
// Go runtime expects and so the symbol it calls needs to be exported
// for external linking to work.
//go:cgo_export_static _cgo_reginit
