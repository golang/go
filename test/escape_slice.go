// errorcheck -0 -m -l

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for slices.

package escape

import (
	"os"
	"strings"
)

var sink interface{}

func slice0() {
	var s []*int
	// BAD: i should not escape
	i := 0 // ERROR "moved to heap: i"
	s = append(s, &i)
	_ = s
}

func slice1() *int {
	var s []*int
	i := 0 // ERROR "moved to heap: i"
	s = append(s, &i)
	return s[0]
}

func slice2() []*int {
	var s []*int
	i := 0 // ERROR "moved to heap: i"
	s = append(s, &i)
	return s
}

func slice3() *int {
	var s []*int
	i := 0 // ERROR "moved to heap: i"
	s = append(s, &i)
	for _, p := range s {
		return p
	}
	return nil
}

func slice4(s []*int) { // ERROR "s does not escape"
	i := 0 // ERROR "moved to heap: i"
	s[0] = &i
}

func slice5(s []*int) { // ERROR "s does not escape"
	if s != nil {
		s = make([]*int, 10) // ERROR "make\(\[\]\*int, 10\) does not escape"
	}
	i := 0 // ERROR "moved to heap: i"
	s[0] = &i
}

func slice6() {
	s := make([]*int, 10) // ERROR "make\(\[\]\*int, 10\) does not escape"
	// BAD: i should not escape
	i := 0 // ERROR "moved to heap: i"
	s[0] = &i
	_ = s
}

func slice7() *int {
	s := make([]*int, 10) // ERROR "make\(\[\]\*int, 10\) does not escape"
	i := 0                // ERROR "moved to heap: i"
	s[0] = &i
	return s[0]
}

func slice8() {
	i := 0
	s := []*int{&i} // ERROR "\[\]\*int{...} does not escape"
	_ = s
}

func slice9() *int {
	i := 0          // ERROR "moved to heap: i"
	s := []*int{&i} // ERROR "\[\]\*int{...} does not escape"
	return s[0]
}

func slice10() []*int {
	i := 0          // ERROR "moved to heap: i"
	s := []*int{&i} // ERROR "\[\]\*int{...} escapes to heap"
	return s
}

func slice11() {
	i := 2
	s := make([]int, 2, 3) // ERROR "make\(\[\]int, 2, 3\) does not escape"
	s = make([]int, i, 3)  // ERROR "make\(\[\]int, i, 3\) does not escape"
	s = make([]int, i, 1)  // ERROR "make\(\[\]int, i, 1\) does not escape"
	_ = s
}

func slice12(x []int) *[1]int { // ERROR "leaking param: x to result ~r0 level=0$"
	return (*[1]int)(x)
}

func slice13(x []*int) [1]*int { // ERROR "leaking param: x to result ~r0 level=1$"
	return [1]*int(x)
}

func envForDir(dir string) []string { // ERROR "dir does not escape"
	env := os.Environ()
	return mergeEnvLists([]string{"PWD=" + dir}, env) // ERROR ".PWD=. \+ dir escapes to heap" "\[\]string{...} does not escape"
}

func mergeEnvLists(in, out []string) []string { // ERROR "leaking param content: in" "leaking param content: out" "leaking param: out to result ~r0 level=0"
NextVar:
	for _, inkv := range in {
		k := strings.SplitAfterN(inkv, "=", 2)[0]
		for i, outkv := range out {
			if strings.HasPrefix(outkv, k) {
				out[i] = inkv
				continue NextVar
			}
		}
		out = append(out, inkv)
	}
	return out
}

const (
	IPv4len = 4
	IPv6len = 16
)

var v4InV6Prefix = []byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff}

func IPv4(a, b, c, d byte) IP {
	p := make(IP, IPv6len) // ERROR "make\(IP, 16\) escapes to heap"
	copy(p, v4InV6Prefix)
	p[12] = a
	p[13] = b
	p[14] = c
	p[15] = d
	return p
}

type IP []byte

type IPAddr struct {
	IP   IP
	Zone string // IPv6 scoped addressing zone
}

type resolveIPAddrTest struct {
	network       string
	litAddrOrName string
	addr          *IPAddr
	err           error
}

var resolveIPAddrTests = []resolveIPAddrTest{
	{"ip", "127.0.0.1", &IPAddr{IP: IPv4(127, 0, 0, 1)}, nil},
	{"ip4", "127.0.0.1", &IPAddr{IP: IPv4(127, 0, 0, 1)}, nil},
	{"ip4:icmp", "127.0.0.1", &IPAddr{IP: IPv4(127, 0, 0, 1)}, nil},
}

func setupTestData() {
	resolveIPAddrTests = append(resolveIPAddrTests,
		[]resolveIPAddrTest{ // ERROR "\[\]resolveIPAddrTest{...} does not escape"
			{"ip",
				"localhost",
				&IPAddr{IP: IPv4(127, 0, 0, 1)}, // ERROR "&IPAddr{...} escapes to heap"
				nil},
			{"ip4",
				"localhost",
				&IPAddr{IP: IPv4(127, 0, 0, 1)}, // ERROR "&IPAddr{...} escapes to heap"
				nil},
		}...)
}
