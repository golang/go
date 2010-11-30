// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

/*
 * Must match iface.c:/Itable and compilers.
 * NOTE: type.go has an Itable, that is the version of Itab used by the reflection code.
 */
type itab struct {
	Itype  *Type
	Type   *Type
	link   *itab
	bad    int32
	unused int32
	Fn     func() // TODO: [0]func()
}
