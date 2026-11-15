// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testdir_test

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

var logosGoTestModule = []byte("module logostest\n\ngo 1.27\n")

func (t test) runLogosGoIsolated(c logosCase) error {
	c = pinTargetFlags(c)
	base := path.Base(t.goFile)
	tempDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(tempDir, base), c.src, 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tempDir, "go.mod"), logosGoTestModule, 0644); err != nil {
		t.Fatal(err)
	}
	runcmd := func(args ...string) ([]byte, error) {
		cmd := exec.Command(args[0], args[1:]...)
		var buf bytes.Buffer
		cmd.Stdout = &buf
		cmd.Stderr = &buf
		cmd.Env = append(os.Environ(), "GOENV=off", "GOFLAGS=")
		cmd.Dir = tempDir
		cmd.Env = append(cmd.Env, "PWD="+tempDir)
		cmd.Env = append(cmd.Env, "STDLIB_IMPORTCFG="+stdlibImportcfgFile())
		cmd.Env = append(cmd.Env, c.runenv...)
		var err error
		if c.tim != 0 {
			err = cmd.Start()
			if err == nil {
				tick := time.NewTimer(time.Duration(c.tim) * time.Second)
				done := make(chan error)
				go func() {
					done <- cmd.Wait()
				}()
				select {
				case err = <-done:
				case <-tick.C:
					cmd.Process.Signal(os.Interrupt)
					time.Sleep(1 * time.Second)
					cmd.Process.Kill()
					<-done
					err = errTimeout
				}
				tick.Stop()
			}
		} else {
			err = cmd.Run()
		}
		if err != nil && err != errTimeout {
			err = fmt.Errorf("%s\n%s", err, buf.Bytes())
		}
		return buf.Bytes(), err
	}
	buildFlags, optFlags := splitOptFlags(c.flags)
	if c.action == "asmcheck" {
		return t.runLogosGoAsmcheckIsolated(c)
	}
	if len(optFlags) > 0 && c.action == "run" {
		cmd := logosGoBuildCmd(t.goGcflags(), filepath.Join(tempDir, base), buildFlags, optFlags)
		if _, err := runcmd(cmd...); err != nil {
			return err
		}
		out, err := runcmd(append([]string{"./a.exe"}, c.args...)...)
		if err != nil {
			return err
		}
		return t.checkExpectedOutput(out)
	}
	return t.runLogosGo(c, base, tempDir, runcmd)
}

func (t test) runLogosGoAsmcheckIsolated(c logosCase) error {
	buildFlags, optFlags := splitOptFlags(c.flags)
	long := filepath.Join(t.gorootTestDir, t.goFileName())
	ops := t.wantedAsmOpcodes(long)
	self := runtime.GOOS + "/" + runtime.GOARCH
	var lastErr error
	for _, env := range ops.Envs() {
		if string(env) != self && !strings.HasPrefix(string(env), self+"/") && !*allCodegen {
			continue
		}
		if len(optFlags) > 0 {
			if err := t.runLogosGoAsmcheckOpt(c, long, env, buildFlags, optFlags, ops[env]); err != nil {
				lastErr = err
				t.Log(err)
			}
			continue
		}
		cmdline := []string{"build", "-gcflags", "-S=2"}
		for i := 0; i < len(buildFlags); i++ {
			flag := buildFlags[i]
			if v, ok := strings.CutPrefix(flag, "-gcflags="); ok {
				cmdline[2] += " " + v
			} else if v, ok := strings.CutPrefix(flag, "--gcflags="); ok {
				cmdline[2] += " " + v
			} else if flag == "-gcflags" || flag == "--gcflags" {
				i++
				if i < len(buildFlags) {
					cmdline[2] += " " + buildFlags[i]
				}
			} else {
				cmdline = append(cmdline, flag)
			}
		}
		cmdline = append(cmdline, "-o", filepath.Join(t.TempDir(), "a.exe"), long)
		cmd := exec.Command(goTool, cmdline...)
		cmd.Env = logosAsmcheckEnv(c, env, buildFlags)
		var buf bytes.Buffer
		cmd.Stdout, cmd.Stderr = &buf, &buf
		if err := cmd.Run(); err != nil {
			lastErr = err
			t.Log(env, "\n", cmd.Stderr)
		}
		if err := t.asmCheck(buf.String(), long, env, ops[env]); err != nil {
			lastErr = err
			t.Log(err)
		}
	}
	if lastErr != nil {
		return errors.New("One or more asmcheck tests failed. Check log for failure details.")
	}
	return nil
}
