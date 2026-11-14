// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the snapshot semantics of range string([]byte). Issue 27148.
package main

import "unsafe"

func checksum(p []byte) (count, sum int) {
	for i, r := range string(p) {
		count++
		sum += i + int(r)
	}
	return
}

func control(p []byte) int {
	n := 0
	for i, r := range string(p) {
		if r == 0 {
			break
		}
		if i&1 == 0 {
			continue
		}
		n += int(r)
	}
	return n
}

type rangeBytes []byte
type rangeText string

func nested(p rangeBytes) int {
	n, m := 0, 1
	for _, r := range rangeText(p) {
		for j := 0; j < 2; j++ {
			n, m = m, n+int(r)
		}
	}
	return n + m
}

func snapshot(p, alias []byte) int {
	n := 0
	for _, r := range string(p) {
		alias[len(alias)-1] = 'z'
		n += int(r)
	}
	return n
}

//go:noinline
func mutate(p []byte) {
	p[len(p)-1] = 'z'
}

func call(p []byte) int {
	n := 0
	for _, r := range string(p) {
		mutate(p)
		n += int(r)
	}
	return n
}

func callback(p []byte, f func()) int {
	n := 0
	for _, r := range string(p) {
		f()
		n += int(r)
	}
	return n
}

func callbackInit(p []byte, f func() int) int {
	n := 0
	for _, r := range string(p) {
		if delta := f(); delta != 0 {
			n += delta
		}
		n = n + int(r) + f()
	}
	return n
}

var global uint32

func globalAlias() int {
	global = 0x61616161
	p := unsafe.Slice((*byte)(unsafe.Pointer(&global)), 4)
	n := 0
	for _, r := range string(p) {
		global = 0
		n += int(r)
	}
	return n
}

func addressTaken() int {
	word := uint32(0x61616161)
	p := unsafe.Slice((*byte)(unsafe.Pointer(&word)), 4)
	n := 0
	for _, r := range string(p) {
		word = 0
		n += int(r)
	}
	return n
}

func rangeValue() int {
	word := rune(0x61616161)
	p := unsafe.Slice((*byte)(unsafe.Pointer(&word)), 4)
	n := 0
	for _, word = range string(p) {
		n += int(word)
	}
	return n
}

func rangeKey() int {
	word := int(0x61616161)
	p := unsafe.Slice((*byte)(unsafe.Pointer(&word)), int(unsafe.Sizeof(word)))
	for i := range p {
		p[i] = 'a'
	}
	n := 0
	var r rune
	for word, r = range string(p) {
		n += int(r)
	}
	return n
}

var sourceCalls int

//go:noinline
func source(p []byte) []byte {
	sourceCalls++
	return p
}

func evaluation(p []byte) int {
	n := 0
	for _, r := range string(source(p)) {
		n += int(r)
	}
	return n
}

func panicking(p []byte) (n int) {
	defer func() {
		if recover() == nil {
			panic("missing bounds panic")
		}
	}()
	for i, r := range string(p) {
		n += int(r)
		_ = p[i+1]
	}
	return
}

func main() {
	for _, p := range [][]byte{nil, {}, []byte("abc"), []byte("aé世😀"), {0xff, 0xc0, 0x80, 0xe2, 0x82}, []byte("a\x00b")} {
		wantCount, wantSum, wantControl := 0, 0, 0
		s := string(p)
		for i, r := range s {
			wantCount++
			wantSum += i + int(r)
		}
		for i, r := range s {
			if r == 0 {
				break
			}
			if i&1 != 0 {
				wantControl += int(r)
			}
		}
		n, m := 0, 1
		for _, r := range s {
			for j := 0; j < 2; j++ {
				n, m = m, n+int(r)
			}
		}
		if count, sum := checksum(p); count != wantCount || sum != wantSum {
			panic("incorrect UTF-8 iteration")
		}
		if control(p) != wantControl {
			panic("incorrect break/continue")
		}
		if nested(p) != n+m {
			panic("incorrect nested loop or multiple assignment")
		}
	}
	want := int('a' + 'b' + 'c')
	p := []byte("abc")
	if snapshot(p, p) != want || string(p) != "abz" {
		panic("source mutation changed snapshot")
	}
	p = []byte("abc")
	if call(p) != want || string(p) != "abz" {
		panic("call mutation changed snapshot")
	}
	p = []byte("abc")
	if callback(p, func() { p[2] = 'z' }) != want || string(p) != "abz" {
		panic("callback mutation changed snapshot")
	}
	p = []byte("abc")
	if callbackInit(p, func() int {
		p[2] = 'z'
		return 0
	}) != want || string(p) != "abz" {
		panic("init or assignment callback changed snapshot")
	}
	if globalAlias() != 4*'a' {
		panic("global mutation changed snapshot")
	}
	if addressTaken() != 4*'a' || rangeValue() != 4*'a' || rangeKey() != int(unsafe.Sizeof(int(0)))*'a' {
		panic("aliased local or range assignment changed snapshot")
	}
	if evaluation([]byte("abc")) != want || sourceCalls != 1 {
		panic("range expression evaluated incorrectly")
	}
	if panicking([]byte("abc")) != want {
		panic("panic lost an earlier side effect")
	}
}
