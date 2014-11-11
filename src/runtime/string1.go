// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

//go:nosplit
func findnull(s *byte) int {
	if s == nil {
		return 0
	}
	p := (*[_MaxMem/2 - 1]byte)(unsafe.Pointer(s))
	l := 0
	for p[l] != 0 {
		l++
	}
	return l
}

func findnullw(s *uint16) int {
	if s == nil {
		return 0
	}
	p := (*[_MaxMem/2/2 - 1]uint16)(unsafe.Pointer(s))
	l := 0
	for p[l] != 0 {
		l++
	}
	return l
}

var maxstring uintptr = 256 // a hint for print

//go:nosplit
func gostringnocopy(str *byte) string {
	var s string
	sp := (*stringStruct)(unsafe.Pointer(&s))
	sp.str = unsafe.Pointer(str)
	sp.len = findnull(str)
	for {
		ms := maxstring
		if uintptr(len(s)) <= ms || casuintptr(&maxstring, ms, uintptr(len(s))) {
			break
		}
	}
	return s
}

func gostringw(strw *uint16) string {
	var buf [8]byte
	str := (*[_MaxMem/2/2 - 1]uint16)(unsafe.Pointer(strw))
	n1 := 0
	for i := 0; str[i] != 0; i++ {
		n1 += runetochar(buf[:], rune(str[i]))
	}
	s, b := rawstring(n1 + 4)
	n2 := 0
	for i := 0; str[i] != 0; i++ {
		// check for race
		if n2 >= n1 {
			break
		}
		n2 += runetochar(b[n2:], rune(str[i]))
	}
	b[n2] = 0 // for luck
	return s[:n2]
}

func strcmp(s1, s2 *byte) int32 {
	p1 := (*[_MaxMem/2 - 1]byte)(unsafe.Pointer(s1))
	p2 := (*[_MaxMem/2 - 1]byte)(unsafe.Pointer(s2))

	for i := uintptr(0); ; i++ {
		c1 := p1[i]
		c2 := p2[i]
		if c1 < c2 {
			return -1
		}
		if c1 > c2 {
			return +1
		}
		if c1 == 0 {
			return 0
		}
	}
}

func strncmp(s1, s2 *byte, n uintptr) int32 {
	p1 := (*[_MaxMem/2 - 1]byte)(unsafe.Pointer(s1))
	p2 := (*[_MaxMem/2 - 1]byte)(unsafe.Pointer(s2))

	for i := uintptr(0); i < n; i++ {
		c1 := p1[i]
		c2 := p2[i]
		if c1 < c2 {
			return -1
		}
		if c1 > c2 {
			return +1
		}
		if c1 == 0 {
			break
		}
	}
	return 0
}
