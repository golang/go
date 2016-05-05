// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build openbsd

package cgo

import _ "unsafe" // for go:linkname

// Supply environ, __progname and __guard_local, because
// we don't link against the standard OpenBSD crt0.o and
// the libc dynamic library needs them.

//go:linkname _environ environ
//go:linkname _progname __progname
//go:linkname _guard_local __guard_local

var _environ uintptr
var _progname uintptr
var _guard_local uintptr

//go:cgo_export_dynamic environ environ
//go:cgo_export_dynamic __progname __progname

// This is normally marked as hidden and placed in the
// .openbsd.randomdata section.
//go:cgo_export_dynamic __guard_local __guard_local

// We override pthread_create to support PT_TLS.
//go:cgo_export_dynamic pthread_create pthread_create
