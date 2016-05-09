// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"debug/elf"
	"debug/macho"
	"encoding/binary"
	"internal/testenv"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

var lldbPath string

func checkLldbPython(t *testing.T) {
	cmd := exec.Command("lldb", "-P")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Skipf("skipping due to issue running lldb: %v\n%s", err, out)
	}
	lldbPath = strings.TrimSpace(string(out))

	cmd = exec.Command("/usr/bin/python2.7", "-c", "import sys;sys.path.append(sys.argv[1]);import lldb; print('go lldb python support')", lldbPath)
	out, err = cmd.CombinedOutput()

	if err != nil {
		t.Skipf("skipping due to issue running python: %v\n%s", err, out)
	}
	if string(out) != "go lldb python support\n" {
		t.Skipf("skipping due to lack of python lldb support: %s", out)
	}

	if runtime.GOOS == "darwin" {
		// Try to see if we have debugging permissions.
		cmd = exec.Command("/usr/sbin/DevToolsSecurity", "-status")
		out, err = cmd.CombinedOutput()
		if err != nil {
			t.Skipf("DevToolsSecurity failed: %v", err)
		} else if !strings.Contains(string(out), "enabled") {
			t.Skip(string(out))
		}
		cmd = exec.Command("/usr/bin/groups")
		out, err = cmd.CombinedOutput()
		if err != nil {
			t.Skipf("groups failed: %v", err)
		} else if !strings.Contains(string(out), "_developer") {
			t.Skip("Not in _developer group")
		}
	}
}

const lldbHelloSource = `
package main
import "fmt"
func main() {
	mapvar := make(map[string]string,5)
	mapvar["abc"] = "def"
	mapvar["ghi"] = "jkl"
	intvar := 42
	ptrvar := &intvar
	fmt.Println("hi") // line 10
	_ = ptrvar
}
`

const lldbScriptSource = `
import sys
sys.path.append(sys.argv[1])
import lldb
import os

TIMEOUT_SECS = 5

debugger = lldb.SBDebugger.Create()
debugger.SetAsync(True)
target = debugger.CreateTargetWithFileAndArch("a.exe", None)
if target:
  print "Created target"
  main_bp = target.BreakpointCreateByLocation("main.go", 10)
  if main_bp:
    print "Created breakpoint"
  process = target.LaunchSimple(None, None, os.getcwd())
  if process:
    print "Process launched"
    listener = debugger.GetListener()
    process.broadcaster.AddListener(listener, lldb.SBProcess.eBroadcastBitStateChanged)
    while True:
      event = lldb.SBEvent()
      if listener.WaitForEvent(TIMEOUT_SECS, event):
        if lldb.SBProcess.GetRestartedFromEvent(event):
          continue
        state = process.GetState()
        if state in [lldb.eStateUnloaded, lldb.eStateLaunching, lldb.eStateRunning]:
          continue
      else:
        print "Timeout launching"
      break
    if state == lldb.eStateStopped:
      for t in process.threads:
        if t.GetStopReason() == lldb.eStopReasonBreakpoint:
          print "Hit breakpoint"
          frame = t.GetFrameAtIndex(0)
          if frame:
            if frame.line_entry:
              print "Stopped at %s:%d" % (frame.line_entry.file.basename, frame.line_entry.line)
            if frame.function:
              print "Stopped in %s" % (frame.function.name,)
            var = frame.FindVariable('intvar')
            if var:
              print "intvar = %s" % (var.GetValue(),)
            else:
              print "no intvar"
    else:
      print "Process state", state
    process.Destroy()
else:
  print "Failed to create target a.exe"

lldb.SBDebugger.Destroy(debugger)
sys.exit()
`

const expectedLldbOutput = `Created target
Created breakpoint
Process launched
Hit breakpoint
Stopped at main.go:10
Stopped in main.main
intvar = 42
`

func TestLldbPython(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	if final := os.Getenv("GOROOT_FINAL"); final != "" && runtime.GOROOT() != final {
		t.Skip("gdb test can fail with GOROOT_FINAL pending")
	}

	checkLldbPython(t)

	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	src := filepath.Join(dir, "main.go")
	err = ioutil.WriteFile(src, []byte(lldbHelloSource), 0644)
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}

	cmd := exec.Command("go", "build", "-gcflags", "-N -l", "-o", "a.exe")
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("building source %v\n%s", err, out)
	}

	src = filepath.Join(dir, "script.py")
	err = ioutil.WriteFile(src, []byte(lldbScriptSource), 0755)
	if err != nil {
		t.Fatalf("failed to create script: %v", err)
	}

	cmd = exec.Command("/usr/bin/python2.7", "script.py", lldbPath)
	cmd.Dir = dir
	got, _ := cmd.CombinedOutput()

	if string(got) != expectedLldbOutput {
		if strings.Contains(string(got), "Timeout launching") {
			t.Skip("Timeout launching")
		}
		t.Fatalf("Unexpected lldb output:\n%s", got)
	}
}

// Check that aranges are valid even when lldb isn't installed.
func TestDwarfAranges(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	src := filepath.Join(dir, "main.go")
	err = ioutil.WriteFile(src, []byte(lldbHelloSource), 0644)
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}

	cmd := exec.Command("go", "build", "-o", "a.exe")
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("building source %v\n%s", err, out)
	}

	filename := filepath.Join(dir, "a.exe")
	if f, err := elf.Open(filename); err == nil {
		sect := f.Section(".debug_aranges")
		if sect == nil {
			t.Fatal("Missing aranges section")
		}
		verifyAranges(t, f.ByteOrder, sect.Open())
	} else if f, err := macho.Open(filename); err == nil {
		sect := f.Section("__debug_aranges")
		if sect == nil {
			t.Fatal("Missing aranges section")
		}
		verifyAranges(t, f.ByteOrder, sect.Open())
	} else {
		t.Skip("Not an elf or macho binary.")
	}
}

func verifyAranges(t *testing.T, byteorder binary.ByteOrder, data io.ReadSeeker) {
	var header struct {
		UnitLength  uint32 // does not include the UnitLength field
		Version     uint16
		Offset      uint32
		AddressSize uint8
		SegmentSize uint8
	}
	for {
		offset, err := data.Seek(0, io.SeekCurrent)
		if err != nil {
			t.Fatalf("Seek error: %v", err)
		}
		if err = binary.Read(data, byteorder, &header); err == io.EOF {
			return
		} else if err != nil {
			t.Fatalf("Error reading arange header: %v", err)
		}
		tupleSize := int64(header.SegmentSize) + 2*int64(header.AddressSize)
		lastTupleOffset := offset + int64(header.UnitLength) + 4 - tupleSize
		if lastTupleOffset%tupleSize != 0 {
			t.Fatalf("Invalid arange length %d, (addr %d, seg %d)", header.UnitLength, header.AddressSize, header.SegmentSize)
		}
		if _, err = data.Seek(lastTupleOffset, io.SeekStart); err != nil {
			t.Fatalf("Seek error: %v", err)
		}
		buf := make([]byte, tupleSize)
		if n, err := data.Read(buf); err != nil || int64(n) < tupleSize {
			t.Fatalf("Read error: %v", err)
		}
		for _, val := range buf {
			if val != 0 {
				t.Fatalf("Invalid terminator")
			}
		}
	}
}
