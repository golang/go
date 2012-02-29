// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof_test

import (
	"bytes"
	"hash/crc32"
	"os/exec"
	"runtime"
	. "runtime/pprof"
	"strings"
	"testing"
	"unsafe"
)

func TestCPUProfile(t *testing.T) {
	switch runtime.GOOS {
	case "darwin":
		out, err := exec.Command("uname", "-a").CombinedOutput()
		if err != nil {
			t.Fatal(err)
		}
		vers := string(out)
		t.Logf("uname -a: %v", vers)
		if strings.Contains(vers, "Darwin Kernel Version 10.8.0") && strings.Contains(vers, "root:xnu-1504.15.3~1/RELEASE_X86_64") {
			t.Logf("skipping test on known-broken kernel (64-bit Snow Leopard)")
			return
		}
	case "plan9":
		// unimplemented
		return
	}

	buf := make([]byte, 100000)
	var prof bytes.Buffer
	if err := StartCPUProfile(&prof); err != nil {
		t.Fatal(err)
	}
	// This loop takes about a quarter second on a 2 GHz laptop.
	// We only need to get one 100 Hz clock tick, so we've got
	// a 25x safety buffer.
	for i := 0; i < 1000; i++ {
		crc32.ChecksumIEEE(buf)
	}
	StopCPUProfile()

	// Convert []byte to []uintptr.
	bytes := prof.Bytes()
	val := *(*[]uintptr)(unsafe.Pointer(&bytes))
	val = val[:len(bytes)/int(unsafe.Sizeof(uintptr(0)))]

	if len(val) < 10 {
		t.Fatalf("profile too short: %#x", val)
	}
	if val[0] != 0 || val[1] != 3 || val[2] != 0 || val[3] != 1e6/100 || val[4] != 0 {
		t.Fatalf("unexpected header %#x", val[:5])
	}

	// Check that profile is well formed and contains ChecksumIEEE.
	found := false
	val = val[5:]
	for len(val) > 0 {
		if len(val) < 2 || val[0] < 1 || val[1] < 1 || uintptr(len(val)) < 2+val[1] {
			t.Fatalf("malformed profile.  leftover: %#x", val)
		}
		for _, pc := range val[2 : 2+val[1]] {
			f := runtime.FuncForPC(pc)
			if f == nil {
				continue
			}
			if strings.Contains(f.Name(), "ChecksumIEEE") {
				found = true
			}
		}
		val = val[2+val[1]:]
	}

	if !found {
		t.Fatal("did not find ChecksumIEEE in the profile")
	}
}
