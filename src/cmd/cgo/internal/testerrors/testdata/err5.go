// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//line /tmp/_cgo_.go:1
//go:cgo_dynamic_linker "/elf/interp"
// ERROR MESSAGE: only allowed in cgo-generated code

func main() {}
