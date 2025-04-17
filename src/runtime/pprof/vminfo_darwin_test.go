// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !ios

package pprof

import (
	"bufio"
	"bytes"
	"fmt"
	"internal/abi"
	"internal/testenv"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestVMInfo(t *testing.T) {
	var begin, end, offset uint64
	var filename string
	first := true
	machVMInfo(func(lo, hi, off uint64, file, buildID string) {
		if first {
			begin = lo
			end = hi
			offset = off
			filename = file
		}
		// May see multiple text segments if rosetta is used for running
		// the go toolchain itself.
		first = false
	})
	lo, hi, err := useVMMapWithRetry(t)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := begin, lo; got != want {
		t.Errorf("got %x, want %x", got, want)
	}
	if got, want := end, hi; got != want {
		t.Errorf("got %x, want %x", got, want)
	}
	if got, want := offset, uint64(0); got != want {
		t.Errorf("got %x, want %x", got, want)
	}
	if !strings.HasSuffix(filename, "pprof.test") {
		t.Errorf("got %s, want pprof.test", filename)
	}
	addr := uint64(abi.FuncPCABIInternal(TestVMInfo))
	if addr < lo || addr > hi {
		t.Errorf("%x..%x does not contain function %p (%x)", lo, hi, TestVMInfo, addr)
	}
}

type mapping struct {
	hi, lo uint64
	err    error
}

func useVMMapWithRetry(t *testing.T) (hi, lo uint64, err error) {
	var retryable bool
	ch := make(chan mapping)
	go func() {
		for {
			hi, lo, retryable, err = useVMMap(t)
			if err == nil {
				ch <- mapping{hi, lo, nil}
				return
			}
			if !retryable {
				ch <- mapping{0, 0, err}
				return
			}
			t.Logf("retrying vmmap after error: %v", err)
		}
	}()
	select {
	case m := <-ch:
		return m.hi, m.lo, m.err
	case <-time.After(time.Minute):
		t.Skip("vmmap taking too long")
	}
	return 0, 0, fmt.Errorf("unreachable")
}

func useVMMap(t *testing.T) (hi, lo uint64, retryable bool, err error) {
	pid := strconv.Itoa(os.Getpid())
	testenv.MustHaveExecPath(t, "vmmap")
	cmd := testenv.Command(t, "vmmap", pid)
	out, cmdErr := cmd.Output()
	if cmdErr != nil {
		t.Logf("vmmap output: %s", out)
		if ee, ok := cmdErr.(*exec.ExitError); ok && len(ee.Stderr) > 0 {
			t.Logf("%v: %v\n%s", cmd, cmdErr, ee.Stderr)
			if testing.Short() && (strings.Contains(string(ee.Stderr), "No process corpse slots currently available, waiting to get one") || strings.Contains(string(ee.Stderr), "Failed to generate corpse from the process")) {
				t.Skipf("Skipping knwn flake in short test mode")
			}
			retryable = bytes.Contains(ee.Stderr, []byte("resource shortage"))
		}
		t.Logf("%v: %v\n", cmd, cmdErr)
		if retryable {
			return 0, 0, true, cmdErr
		}
	}
	// Always parse the output of vmmap since it may return an error
	// code even if it successfully reports the text segment information
	// required for this test.
	hi, lo, err = parseVmmap(out)
	if err != nil {
		if cmdErr != nil {
			return 0, 0, false, fmt.Errorf("failed to parse vmmap output, vmmap reported an error: %v", err)
		}
		t.Logf("vmmap output: %s", out)
		return 0, 0, false, fmt.Errorf("failed to parse vmmap output, vmmap did not report an error: %v", err)
	}
	return hi, lo, false, nil
}

// parseVmmap parses the output of vmmap and calls addMapping for the first r-x TEXT segment in the output.
func parseVmmap(data []byte) (hi, lo uint64, err error) {
	// vmmap 53799
	// Process:         gopls [53799]
	// Path:            /Users/USER/*/gopls
	// Load Address:    0x1029a0000
	// Identifier:      gopls
	// Version:         ???
	// Code Type:       ARM64
	// Platform:        macOS
	// Parent Process:  Code Helper (Plugin) [53753]
	//
	// Date/Time:       2023-05-25 09:45:49.331 -0700
	// Launch Time:     2023-05-23 09:35:37.514 -0700
	// OS Version:      macOS 13.3.1 (22E261)
	// Report Version:  7
	// Analysis Tool:   /Applications/Xcode.app/Contents/Developer/usr/bin/vmmap
	// Analysis Tool Version:  Xcode 14.3 (14E222b)
	//
	// Physical footprint:         1.2G
	// Physical footprint (peak):  1.2G
	// Idle exit:                  untracked
	// ----
	//
	// Virtual Memory Map of process 53799 (gopls)
	// Output report format:  2.4  -64-bit process
	// VM page size:  16384 bytes
	//
	// ==== Non-writable regions for process 53799
	// REGION TYPE                    START END         [ VSIZE  RSDNT  DIRTY   SWAP] PRT/MAX SHRMOD PURGE    REGION DETAIL
	// __TEXT                      1029a0000-1033bc000    [ 10.1M  7360K     0K     0K] r-x/rwx SM=COW          /Users/USER/*/gopls
	// __DATA_CONST                1033bc000-1035bc000    [ 2048K  2000K     0K     0K] r--/rwSM=COW          /Users/USER/*/gopls
	// __DATA_CONST                1035bc000-103a48000    [ 4656K  3824K     0K     0K] r--/rwSM=COW          /Users/USER/*/gopls
	// __LINKEDIT                  103b00000-103c98000    [ 1632K  1616K     0K     0K] r--/r-SM=COW          /Users/USER/*/gopls
	// dyld private memory         103cd8000-103cdc000    [   16K     0K     0K     0K] ---/--SM=NUL
	// shared memory               103ce4000-103ce8000    [   16K    16K    16K     0K] r--/r-SM=SHM
	// MALLOC metadata             103ce8000-103cec000    [   16K    16K    16K     0K] r--/rwx SM=COW          DefaultMallocZone_0x103ce8000 zone structure
	// MALLOC guard page           103cf0000-103cf4000    [   16K     0K     0K     0K] ---/rwx SM=COW
	// MALLOC guard page           103cfc000-103d00000    [   16K     0K     0K     0K] ---/rwx SM=COW
	// MALLOC guard page           103d00000-103d04000    [   16K     0K     0K     0K] ---/rwx SM=NUL

	banner := "==== Non-writable regions for process"
	grabbing := false
	sc := bufio.NewScanner(bytes.NewReader(data))
	for sc.Scan() {
		l := sc.Text()
		if grabbing {
			p := strings.Fields(l)
			if len(p) > 7 && p[0] == "__TEXT" && p[7] == "r-x/rwx" {
				locs := strings.Split(p[1], "-")
				start, _ := strconv.ParseUint(locs[0], 16, 64)
				end, _ := strconv.ParseUint(locs[1], 16, 64)
				return start, end, nil
			}
		}
		if strings.HasPrefix(l, banner) {
			grabbing = true
		}
	}
	return 0, 0, fmt.Errorf("vmmap no text segment found")
}
