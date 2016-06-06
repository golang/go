// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package route

import (
	"testing"
	"time"
	"unsafe"
)

func TestFetchAndParseRIBOnFreeBSD(t *testing.T) {
	for _, af := range []int{sysAF_UNSPEC, sysAF_INET, sysAF_INET6} {
		for _, typ := range []RIBType{sysNET_RT_IFMALIST} {
			ms, err := fetchAndParseRIB(af, typ)
			if err != nil {
				t.Error(err)
				continue
			}
			ss, err := msgs(ms).validate()
			if err != nil {
				t.Errorf("%v %d %v", addrFamily(af), typ, err)
				continue
			}
			for _, s := range ss {
				t.Log(s)
			}
		}
	}
}

func TestFetchAndParseRIBOnFreeBSD10AndAbove(t *testing.T) {
	if _, err := FetchRIB(sysAF_UNSPEC, sysNET_RT_IFLISTL, 0); err != nil {
		t.Skip("NET_RT_IFLISTL not supported")
	}
	var p uintptr
	if kernelAlign != int(unsafe.Sizeof(p)) {
		t.Skip("NET_RT_IFLIST vs. NET_RT_IFLISTL doesn't work for 386 emulation on amd64")
	}

	var tests = [2]struct {
		typ  RIBType
		b    []byte
		msgs []Message
		ss   []string
	}{
		{typ: sysNET_RT_IFLIST},
		{typ: sysNET_RT_IFLISTL},
	}
	for _, af := range []int{sysAF_UNSPEC, sysAF_INET, sysAF_INET6} {
		var lastErr error
		for i := 0; i < 3; i++ {
			for j := range tests {
				var err error
				if tests[j].b, err = FetchRIB(af, tests[j].typ, 0); err != nil {
					lastErr = err
					time.Sleep(10 * time.Millisecond)
				}
			}
			if lastErr == nil {
				break
			}
		}
		if lastErr != nil {
			t.Error(af, lastErr)
			continue
		}
		for i := range tests {
			var err error
			if tests[i].msgs, err = ParseRIB(tests[i].typ, tests[i].b); err != nil {
				lastErr = err
				t.Error(af, err)
			}
		}
		if lastErr != nil {
			continue
		}
		for i := range tests {
			var err error
			tests[i].ss, err = msgs(tests[i].msgs).validate()
			if err != nil {
				lastErr = err
				t.Error(af, err)
			}
			for _, s := range tests[i].ss {
				t.Log(s)
			}
		}
		if lastErr != nil {
			continue
		}
		for i := len(tests) - 1; i > 0; i-- {
			if len(tests[i].ss) != len(tests[i-1].ss) {
				t.Errorf("got %v; want %v", tests[i].ss, tests[i-1].ss)
				continue
			}
			for j, s1 := range tests[i].ss {
				s0 := tests[i-1].ss[j]
				if s1 != s0 {
					t.Errorf("got %s; want %s", s1, s0)
				}
			}
		}
	}
}
