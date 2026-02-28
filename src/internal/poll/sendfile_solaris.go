// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

//go:cgo_ldflag "-lsendfile"

// Not strictly needed, but very helpful for debugging, see issue #10221.
//
//go:cgo_import_dynamic _ _ "libsendfile.so"
//go:cgo_import_dynamic _ _ "libsocket.so"
