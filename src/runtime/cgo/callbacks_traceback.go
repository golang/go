// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin linux

package cgo

import _ "unsafe" // for go:linkname

// Calls the traceback function passed to SetCgoTraceback.

//go:cgo_import_static x_cgo_callers
//go:linkname x_cgo_callers x_cgo_callers
//go:linkname _cgo_callers _cgo_callers
var x_cgo_callers byte
var _cgo_callers = &x_cgo_callers
