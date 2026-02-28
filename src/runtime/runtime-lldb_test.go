// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"internal/testenv"
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
	testenv.SkipFlaky(t, 31188)

	checkLldbPython(t)

	dir := t.TempDir()

	src := filepath.Join(dir, "main.go")
	err := os.WriteFile(src, []byte(lldbHelloSource), 0644)
	if err != nil {
		t.Fatalf("failed to create src file: %v", err)
	}

	mod := filepath.Join(dir, "go.mod")
	err = os.WriteFile(mod, []byte("module lldbtest"), 0644)
	if err != nil {
		t.Fatalf("failed to create mod file: %v", err)
	}

	// As of 2018-07-17, lldb doesn't support compressed DWARF, so
	// disable it for this test.
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-gcflags=all=-N -l", "-ldflags=-compressdwarf=false", "-o", "a.exe")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOPATH=") // issue 31100
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("building source %v\n%s", err, out)
	}

	src = filepath.Join(dir, "script.py")
	err = os.WriteFile(src, []byte(lldbScriptSource), 0755)
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
