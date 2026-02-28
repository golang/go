// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (386 || amd64 || amd64p32) && gccgo

package cpu

//extern gccgoGetCpuidCount
func gccgoGetCpuidCount(eaxArg, ecxArg uint32, eax, ebx, ecx, edx *uint32)

func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32) {
	var a, b, c, d uint32
	gccgoGetCpuidCount(eaxArg, ecxArg, &a, &b, &c, &d)
	return a, b, c, d
}

//extern gccgoXgetbv
func gccgoXgetbv(eax, edx *uint32)

func xgetbv() (eax, edx uint32) {
	var a, d uint32
	gccgoXgetbv(&a, &d)
	return a, d
}
