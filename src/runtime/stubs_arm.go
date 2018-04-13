// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// Stubs to pacify vet. Not safe to call from Go.
// Calls to these functions are inserted by the compiler or assembler.
func udiv()
func _div()
func _divu()
func _mod()
func _modu()
