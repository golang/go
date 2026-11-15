// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testdir_test

import (
	"bytes"
	"errors"
	"fmt"
	"go/scanner"
	"go/token"
	"internal/testenv"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
)

func testFilesForDir(t *testing.T, dir string) []string {
	if dir != "logos" {
		return goFiles(t, dir)
	}
	root := filepath.Join(testenv.GOROOT(t), "test", "logos")
	var names []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || strings.HasPrefix(d.Name(), ".") {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if strings.HasSuffix(rel, ".go") || strings.HasSuffix(rel, ".lgo") {
			if shardMatch(rel) {
				names = append(names, rel)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	slices.Sort(names)
	return names
}

func logosToolPath(t *testing.T) string {
	t.Helper()
	name := "logos"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	p := filepath.Join(filepath.Dir(goTool), name)
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("logos binary not found at %s: %v (rebuild the toolchain with make.bash)", p, err)
	}
	return p
}

func testOutputStem(filename string) string {
	if stem, ok := strings.CutSuffix(filename, ".lgo"); ok {
		return stem
	}
	if stem, ok := strings.CutSuffix(filename, ".go"); ok {
		return stem
	}
	return filename
}

func lgoSourceHasMainFunc(src []byte) bool {
	var s scanner.Scanner
	fset := token.NewFileSet()
	f := fset.AddFile("", fset.Base(), len(src))
	s.Init(f, src, nil, 0)

	isPkgMain := false
	sawPackage := false
	depth := 0
	prevFunc := false

	for {
		_, tok, lit := s.Scan()
		if tok == token.EOF {
			return false
		}
		switch tok {
		case token.PACKAGE:
			sawPackage = true
			continue
		case token.LBRACE:
			depth++
			prevFunc = false
			continue
		case token.RBRACE:
			if depth > 0 {
				depth--
			}
			prevFunc = false
			continue
		case token.FUNC:
			if depth == 0 && isPkgMain {
				prevFunc = true
			}
			continue
		case token.LPAREN:
			prevFunc = false
			continue
		case token.IDENT:
			if sawPackage && !isPkgMain {
				if lit == "main" {
					isPkgMain = true
				}
				sawPackage = false
				continue
			}
			if prevFunc && lit == "main" {
				return true
			}
			prevFunc = false
			continue
		}
		sawPackage = false
		prevFunc = false
	}
}

func routeLogosHarness(dir string, _ string, action string, _ []string) bool {
	if dir != "logos" {
		return false
	}
	if action == "skip" {
		return false
	}
	return true
}

func logosRecipeVariants(dir, src, action string) []string {
	if dir != "logos" {
		return nil
	}
	verb, _, _ := strings.Cut(action, " ")
	_, rest, found := strings.Cut(src, "// "+action+"\n")
	if !found {
		return nil
	}
	var variants []string
	for {
		line, next, _ := strings.Cut(rest, "\n")
		recipe, ok := strings.CutPrefix(line, "// ")
		if !ok || (recipe != verb && !strings.HasPrefix(recipe, verb+" ")) {
			return variants
		}
		variants = append(variants, strings.TrimSpace(recipe))
		rest = next
	}
}

type logosCase struct {
	action    string
	args      []string
	flags     []string
	runenv    []string
	src       []byte
	auto      bool
	wantError bool
	tim       int
}

func pinTargetFlags(c logosCase) logosCase {
	var flags []string
	for _, f := range c.flags {
		if level, ok := strings.CutPrefix(f, "-goamd64="); ok {
			c.runenv = append(c.runenv, "GOAMD64="+level)
			continue
		}
		flags = append(flags, f)
	}
	c.flags = flags
	return c
}

func splitArenaFlag(flags []string) (rest []string, toml []byte) {
	for _, f := range flags {
		if v, ok := strings.CutPrefix(f, "-arena="); ok {
			toml = fmt.Appendf(nil, "[build]\narena = %s\n", v)
			continue
		}
		rest = append(rest, f)
	}
	return rest, toml
}

func splitOptFlags(flags []string) (buildFlags, optFlags []string) {
	for _, f := range flags {
		if strings.HasPrefix(f, "-O=") {
			optFlags = append(optFlags, f)
		} else {
			buildFlags = append(buildFlags, f)
		}
	}
	return buildFlags, optFlags
}

func logosTestFlagsArgs(compilerFlags []string) []string {
	raw := os.Getenv("GO_GCFLAGS")
	first := "-gcflags=all=" + raw
	base := strings.TrimSpace(raw)
	parts := make([]string, 0, len(compilerFlags)+2)
	if base != "" {
		parts = append(parts, base)
	}
	hasE := false
	for _, f := range compilerFlags {
		if f == "-e" {
			hasE = true
			break
		}
	}
	if !hasE {
		parts = append(parts, "-e")
	}
	parts = append(parts, compilerFlags...)
	second := "-gcflags=" + strings.Join(parts, " ")
	return []string{first, second}
}

func logosTestFlags(compilerFlags []string) string {
	return strings.Join(logosTestFlagsArgs(compilerFlags), " ")
}

func withSSACheckOn(flags []string) []string {
	out := slices.Clone(flags)
	for i, f := range out {
		if strings.HasPrefix(f, "-d=") {
			out[i] = f + ",ssa/check/on"
			return out
		}
	}
	return append(out, "-d=ssa/check/on")
}

func logosUnsupportedActionErr(action string) error {
	return fmt.Errorf("logos harness: unsupported action %s for .lgo", action)
}

func withLinkshared(cmd []string) []string {
	if *linkshared {
		return append(cmd, "-linkshared")
	}
	return cmd
}

func logosGoBuildCmd(gcflags, file string, buildFlags, optFlags []string) []string {
	cmd := []string{goTool, "build", gcflags, "-o", "a.exe"}
	if *linkshared {
		cmd = append(cmd, "-linkshared")
	}
	cmd = append(cmd, optFlags...)
	cmd = append(cmd, buildFlags...)
	cmd = append(cmd, file)
	return cmd
}

func (t test) runLogos(c logosCase) error {
	c = pinTargetFlags(c)
	var toml []byte
	c.flags, toml = splitArenaFlag(c.flags)
	base := path.Base(t.goFile)
	isLgo := strings.HasSuffix(base, ".lgo")
	stagedBase := base
	if stem, ok := strings.CutSuffix(base, "_test.lgo"); isLgo && c.action == "run" && ok && lgoSourceHasMainFunc(c.src) {
		stagedBase = stem + "_main.lgo"
	}
	tempDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(tempDir, stagedBase), c.src, 0644); err != nil {
		t.Fatal(err)
	}
	if toml != nil {
		if err := os.WriteFile(filepath.Join(tempDir, "logos.toml"), toml, 0644); err != nil {
			t.Fatal(err)
		}
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
		if flags, _ := splitOptFlags(c.flags); len(flags) > 0 && flags[0] == "-race" {
			cmd.Env = append(cmd.Env, "CGO_ENABLED=1")
		}
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
	if strings.HasPrefix(t.goFile, "lint/") {
		return t.runLogosLint(c, base, tempDir, runcmd)
	}
	if !isLgo {
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
	if err := os.WriteFile(filepath.Join(tempDir, "go.mod"), []byte("module logostest\n\ngo 1.27\n"), 0644); err != nil {
		t.Fatal(err)
	}
	return t.runLogosLgo(c, stagedBase, tempDir, runcmd)
}

func (t test) runLogosLint(c logosCase, base, tempDir string, runcmd runCmd) error {
	if err := os.WriteFile(filepath.Join(tempDir, "go.mod"), []byte("module logostest\n\ngo 1.27\n"), 0644); err != nil {
		t.Fatal(err)
	}
	buildFlags, _ := splitOptFlags(c.flags)
	cmd := append([]string{logosToolPath(t.T), "lint"}, buildFlags...)
	out, err := runcmd(append(cmd, "./...")...)
	if c.wantError && err == nil {
		return fmt.Errorf("logos lint succeeded unexpectedly\n%s", out)
	}
	if !c.wantError && err != nil {
		return err
	}
	full := filepath.Join(tempDir, base)
	return t.errorCheck(string(out), c.auto, full, base, "./"+base, base)
}

func (t test) runLogosLgo(c logosCase, base, tempDir string, runcmd runCmd) error {
	buildFlags, optFlags := splitOptFlags(c.flags)
	switch c.action {
	case "run":
		if strings.HasSuffix(base, "_test.lgo") {
			return t.runLogosLgoTest(buildFlags, runcmd)
		}
		return t.runLogosLgoRun(c, base, tempDir, runcmd)
	case "compile":
		cmd := []string{logosToolPath(t.T), "build"}
		cmd = append(cmd, optFlags...)
		cmd = append(cmd, logosTestFlagsArgs(buildFlags)...)
		cmd = append(cmd, ".")
		_, err := runcmd(cmd...)
		return err
	case "build":
		cmd := []string{logosToolPath(t.T), "build"}
		cmd = append(cmd, buildFlags...)
		cmd = append(cmd, optFlags...)
		cmd = append(cmd, "-o", "a.exe", ".")
		_, err := runcmd(cmd...)
		return err
	case "errorcheck":
		return t.runLogosLgoErrorcheck(c, base, tempDir, runcmd)
	case "runoutput":
		return t.runLogosLgoRunoutput(c, tempDir, runcmd)
	case "asmcheck":
		return t.runLogosLgoAsmcheck(c, base, tempDir)
	default:
		return logosUnsupportedActionErr(c.action)
	}
}

const logosFuzzTimeEnv = "LOGOS_TESTDIR_FUZZTIME"

func logosTestCmd(tool, gcflags string, buildFlags []string, fuzzTime string) []string {
	cmd := []string{tool, "test", "-count=1", gcflags}
	cmd = append(cmd, buildFlags...)
	if fuzzTime != "" {
		cmd = append(cmd, "-fuzz=^Fuzz", "-fuzztime="+fuzzTime)
	}
	return append(cmd, ".")
}

func logosCaseExecCmd(c logosCase) []string {
	var caseGOOS, caseGOARCH string
	for _, kv := range c.runenv {
		if v, ok := strings.CutPrefix(kv, "GOOS="); ok {
			caseGOOS = v
			continue
		}
		if v, ok := strings.CutPrefix(kv, "GOARCH="); ok {
			caseGOARCH = v
		}
	}
	if caseGOOS == "" && caseGOARCH == "" {
		return findExecCmd()
	}
	if caseGOOS == "" {
		caseGOOS = goos
	}
	if caseGOARCH == "" {
		caseGOARCH = goarch
	}
	if caseGOOS == runtime.GOOS && caseGOARCH == runtime.GOARCH {
		return nil
	}
	if p, err := exec.LookPath(fmt.Sprintf("go_%s_%s_exec", caseGOOS, caseGOARCH)); err == nil {
		return []string{p}
	}
	return nil
}

func (t test) runLogosLgoRun(c logosCase, base, tempDir string, runcmd runCmd) error {
	buildFlags, optFlags := splitOptFlags(c.flags)
	buildCmd := []string{logosToolPath(t.T), "build", t.goGcflags()}
	buildCmd = withLinkshared(buildCmd)
	buildCmd = append(buildCmd, buildFlags...)
	buildCmd = append(buildCmd, optFlags...)
	buildCmd = append(buildCmd, "-o", "a.exe", ".")
	buildOut, err := runcmd(buildCmd...)
	if err != nil {
		return err
	}
	if len(buildOut) > 0 {
		t.Logf("logos build output:\n%s", buildOut)
	}
	full := filepath.Join(tempDir, base)
	if len(t.wantedErrors(full, base)) > 0 {
		if err := t.errorCheck(string(buildOut), c.auto, full, base, "./"+base, base); err != nil {
			return err
		}
	}
	exe := filepath.Join(tempDir, "a.exe")
	execCmd := append(slices.Clone(logosCaseExecCmd(c)), exe)
	execCmd = append(execCmd, c.args...)
	out, err := runcmd(execCmd...)
	if err != nil {
		return err
	}
	return t.checkExpectedOutput(out)
}

func (t test) runLogosLgoTest(buildFlags []string, runcmd runCmd) error {
	cmd := logosTestCmd(logosToolPath(t.T), t.goGcflags(), buildFlags, os.Getenv(logosFuzzTimeEnv))
	_, err := runcmd(cmd...)
	return err
}

func (t test) runLogosLgoAsmcheck(c logosCase, base, tempDir string) error {
	buildFlags, optFlags := splitOptFlags(c.flags)
	long := filepath.Join(tempDir, base)
	ops := t.wantedAsmOpcodes(long)
	self := runtime.GOOS + "/" + runtime.GOARCH
	var lastErr error
	for _, env := range ops.Envs() {
		if string(env) != self && !strings.HasPrefix(string(env), self+"/") && !*allCodegen {
			continue
		}
		raw := os.Getenv("GO_GCFLAGS")
		second := "-gcflags="
		if b := strings.TrimSpace(raw); b != "" {
			second += b + " "
		}
		second += "-S=2"
		for _, f := range buildFlags {
			second += " " + f
		}
		cmdline := []string{logosToolPath(t.T), "build"}
		cmdline = append(cmdline, optFlags...)
		cmdline = append(cmdline, "-gcflags=all="+raw, second, ".")
		cmd := exec.Command(cmdline[0], cmdline[1:]...)
		cmd.Dir = tempDir
		cmd.Env = append(os.Environ(), "GOENV=off", "GOFLAGS=", "PWD="+tempDir, "STDLIB_IMPORTCFG="+stdlibImportcfgFile())
		cmd.Env = append(cmd.Env, env.Environ()...)
		cmd.Env = append(cmd.Env, c.runenv...)
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

func (t test) runLogosLgoErrorcheck(c logosCase, base, tempDir string, runcmd runCmd) error {
	buildFlags, optFlags := splitOptFlags(c.flags)
	flags := withSSACheckOn(buildFlags)
	hasPanic := false
	for _, f := range flags {
		if f == "-d=panic" {
			hasPanic = true
			break
		}
	}
	if !hasPanic {
		flags = append([]string{"-d=panic"}, flags...)
	}
	cmd := []string{logosToolPath(t.T), "build"}
	cmd = append(cmd, optFlags...)
	cmd = append(cmd, logosTestFlagsArgs(flags)...)
	cmd = append(cmd, ".")
	out, err := runcmd(cmd...)
	if c.wantError {
		if err == nil {
			return fmt.Errorf("compilation succeeded unexpectedly\n%s", out)
		}
		if err == errTimeout {
			return fmt.Errorf("compilation timed out")
		}
	} else {
		if err != nil {
			return err
		}
	}
	full := filepath.Join(tempDir, base)
	return t.errorCheck(string(out), c.auto, full, base, "./"+base, base)
}

func (t test) runLogosLgoRunoutput(c logosCase, tempDir string, runcmd runCmd) error {
	buildFlags, _ := splitOptFlags(c.flags)
	t.runoutputGate <- true
	defer func() { <-t.runoutputGate }()
	cmd := []string{logosToolPath(t.T), "run", t.goGcflags()}
	cmd = append(cmd, buildFlags...)
	cmd = append(cmd, ".")
	cmd = append(cmd, c.args...)
	out, err := runcmd(cmd...)
	if err != nil {
		return err
	}
	tfile := filepath.Join(tempDir, "tmp__.go")
	if err := os.WriteFile(tfile, out, 0666); err != nil {
		t.Fatalf("write tempfile: %v", err)
	}
	cmd = []string{goTool, "run", t.goGcflags()}
	cmd = withLinkshared(cmd)
	cmd = append(cmd, tfile)
	out, err = runcmd(cmd...)
	if err != nil {
		return err
	}
	return t.checkExpectedOutput(out)
}

func (t test) runLogosGo(c logosCase, base, tempDir string, runcmd runCmd) error {
	buildFlags, _ := splitOptFlags(c.flags)
	full := filepath.Join(tempDir, base)
	switch c.action {
	case "run":
		return t.runLogosGoRun(c, base, tempDir, runcmd)
	case "build":
		cmd := []string{goTool, "build", t.goGcflags()}
		cmd = append(cmd, buildFlags...)
		cmd = append(cmd, "-o", "a.exe", full)
		_, err := runcmd(cmd...)
		return err
	case "compile":
		_, err := compileFile(runcmd, full, buildFlags)
		return err
	case "errorcheck":
		return t.runLogosGoErrorcheck(c, base, tempDir, runcmd)
	case "runoutput":
		return t.runLogosGoRunoutput(c, base, tempDir, runcmd)
	case "asmcheck":
		return t.runLogosGoAsmcheck(c)
	case "buildrun":
		return t.runLogosGoBuildrun(c, base, tempDir, runcmd)
	case "errorcheckoutput":
		return t.runLogosGoErrorcheckOutput(c, base, tempDir, runcmd)
	default:
		return t.runLogosGoDir(c, tempDir, runcmd)
	}
}

func (t test) runLogosGoRun(c logosCase, base, tempDir string, runcmd runCmd) error {
	buildFlags, _ := splitOptFlags(c.flags)
	full := filepath.Join(tempDir, base)
	cmd := []string{goTool, "run", t.goGcflags()}
	cmd = withLinkshared(cmd)
	cmd = append(cmd, buildFlags...)
	cmd = append(cmd, full)
	cmd = append(cmd, c.args...)
	out, err := runcmd(cmd...)
	if err != nil {
		return err
	}
	return t.checkExpectedOutput(out)
}

func (t test) runLogosGoErrorcheck(c logosCase, base, tempDir string, runcmd runCmd) error {
	buildFlags, optFlags := splitOptFlags(c.flags)
	full := filepath.Join(tempDir, base)
	flags := withSSACheckOn(buildFlags)
	if len(optFlags) > 0 {
		return t.runLogosGoErrorcheckOpt(c, base, tempDir, runcmd, flags, optFlags)
	}
	cmdline := []string{goTool, "tool", "compile", "-p=p", "-d=panic", "-C", "-e", "-importcfg=" + stdlibImportcfgFile(), "-o", "a.o"}
	cmdline = append(cmdline, flags...)
	cmdline = append(cmdline, full)
	out, err := runcmd(cmdline...)
	if c.wantError {
		if err == nil {
			return fmt.Errorf("compilation succeeded unexpectedly\n%s", out)
		}
		if err == errTimeout {
			return fmt.Errorf("compilation timed out")
		}
	} else {
		if err != nil {
			return err
		}
	}
	if *updateErrors {
		t.updateErrors(string(out), full)
	}
	return t.errorCheck(string(out), c.auto, full, base)
}

func (t test) runLogosGoErrorcheckOpt(c logosCase, base, tempDir string, runcmd runCmd, flags, optFlags []string) error {
	full := filepath.Join(tempDir, base)
	cmdline := []string{goTool, "build"}
	cmdline = append(cmdline, optFlags...)
	cmdline = append(cmdline, "-gcflags="+strings.Join(flags, " "))
	cmdline = append(cmdline, "-o", "a.exe", full)
	out, err := runcmd(cmdline...)
	if c.wantError {
		if err == nil {
			return fmt.Errorf("compilation succeeded unexpectedly\n%s", out)
		}
		if err == errTimeout {
			return fmt.Errorf("compilation timed out")
		}
	} else {
		if err != nil {
			return err
		}
	}
	if *updateErrors {
		t.updateErrors(string(out), full)
	}
	return t.errorCheck(string(out), c.auto, full, base, "./"+base, base)
}

func (t test) runLogosGoRunoutput(c logosCase, base, tempDir string, runcmd runCmd) error {
	buildFlags, _ := splitOptFlags(c.flags)
	full := filepath.Join(tempDir, base)
	t.runoutputGate <- true
	defer func() { <-t.runoutputGate }()
	cmd := []string{goTool, "run", t.goGcflags()}
	cmd = withLinkshared(cmd)
	cmd = append(cmd, buildFlags...)
	cmd = append(cmd, full)
	cmd = append(cmd, c.args...)
	out, err := runcmd(cmd...)
	if err != nil {
		return err
	}
	tfile := filepath.Join(tempDir, "tmp__.go")
	if err := os.WriteFile(tfile, out, 0666); err != nil {
		t.Fatalf("write tempfile: %v", err)
	}
	cmd = []string{goTool, "run", t.goGcflags()}
	cmd = withLinkshared(cmd)
	cmd = append(cmd, tfile)
	out, err = runcmd(cmd...)
	if err != nil {
		return err
	}
	return t.checkExpectedOutput(out)
}

func logosAsmcheckEnv(c logosCase, env buildEnv, buildFlags []string) []string {
	out := append(os.Environ(), env.Environ()...)
	out = append(out, c.runenv...)
	if len(buildFlags) > 0 && buildFlags[0] == "-race" {
		out = append(out, "CGO_ENABLED=1")
	}
	return out
}

func (t test) runLogosGoAsmcheck(c logosCase) error {
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
		cmdline = append(cmdline, long)
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

func (t test) runLogosGoAsmcheckOpt(c logosCase, long string, env buildEnv, buildFlags, optFlags []string, ops map[string][]wantedAsmOpcode) error {
	gcflags := "-S=2"
	for _, f := range buildFlags {
		gcflags += " " + f
	}
	out := filepath.Join(t.TempDir(), "a.exe")
	cmdline := []string{goTool, "build"}
	cmdline = append(cmdline, optFlags...)
	cmdline = append(cmdline, "-gcflags="+gcflags, "-o", out, long)
	cmd := exec.Command(cmdline[0], cmdline[1:]...)
	cmd.Env = append(os.Environ(), env.Environ()...)
	cmd.Env = append(cmd.Env, c.runenv...)
	var buf bytes.Buffer
	cmd.Stdout, cmd.Stderr = &buf, &buf
	if err := cmd.Run(); err != nil {
		return err
	}
	return t.asmCheck(buf.String(), long, env, ops)
}

func (t test) runLogosGoBuildrun(c logosCase, base, tempDir string, runcmd runCmd) error {
	buildFlags, _ := splitOptFlags(c.flags)
	full := filepath.Join(tempDir, base)
	cmd := []string{goTool, "build", t.goGcflags(), "-o", "a.exe"}
	cmd = withLinkshared(cmd)
	cmd = append(cmd, buildFlags...)
	cmd = append(cmd, full)
	_, err := runcmd(cmd...)
	if err != nil {
		return err
	}
	cmd = []string{"./a.exe"}
	out, err := runcmd(append(cmd, c.args...)...)
	if err != nil {
		return err
	}
	return t.checkExpectedOutput(out)
}

func (t test) runLogosGoErrorcheckOutput(c logosCase, base, tempDir string, runcmd runCmd) error {
	buildFlags, _ := splitOptFlags(c.flags)
	full := filepath.Join(tempDir, base)
	cmd := []string{goTool, "run", t.goGcflags()}
	cmd = withLinkshared(cmd)
	cmd = append(cmd, full)
	out, err := runcmd(append(cmd, c.args...)...)
	if err != nil {
		return err
	}
	tfile := filepath.Join(tempDir, "tmp__.go")
	if err := os.WriteFile(tfile, out, 0666); err != nil {
		t.Fatalf("write tempfile: %v", err)
	}
	cmdline := []string{goTool, "tool", "compile", "-importcfg=" + stdlibImportcfgFile(), "-p=p", "-d=panic", "-e", "-o", "a.o"}
	cmdline = append(cmdline, buildFlags...)
	cmdline = append(cmdline, tfile)
	out, err = runcmd(cmdline...)
	if c.wantError {
		if err == nil {
			return fmt.Errorf("compilation succeeded unexpectedly\n%s", out)
		}
	} else {
		if err != nil {
			return err
		}
	}
	return t.errorCheck(string(out), false, tfile, "tmp__.go")
}

func (t test) runLogosGoDir(c logosCase, tempDir string, runcmd runCmd) error {
	buildFlags, _ := splitOptFlags(c.flags)
	importcfg := func(pkgs []*goDirPkg) string {
		cfg := stdlibImportcfg()
		for _, pkg := range pkgs {
			pkgpath := path.Join("test", strings.TrimSuffix(pkg.files[0], ".go"))
			cfg += "\npackagefile " + pkgpath + "=" + filepath.Join(tempDir, pkgpath+".a")
		}
		filename := filepath.Join(tempDir, "importcfg")
		if err := os.WriteFile(filename, []byte(cfg), 0644); err != nil {
			t.Fatal(err)
		}
		return filename
	}
	switch c.action {
	case "compiledir":
		longdir := filepath.Join(t.gorootTestDir, t.goDirName())
		pkgs := goDirPackages(t.T, longdir, false)
		importcfgfile := importcfg(pkgs)
		for _, pkg := range pkgs {
			_, err := compileInDir(runcmd, longdir, buildFlags, importcfgfile, pkg.name, pkg.files...)
			if err != nil {
				return err
			}
		}
		return nil
	case "errorcheckdir", "errorcheckandrundir":
		flags := append(slices.Clone(buildFlags), "-d=panic")
		longdir := filepath.Join(t.gorootTestDir, t.goDirName())
		pkgs := goDirPackages(t.T, longdir, false)
		errPkg := len(pkgs) - 1
		if c.wantError && c.action == "errorcheckandrundir" {
			errPkg--
		}
		importcfgfile := importcfg(pkgs)
		for i, pkg := range pkgs {
			out, err := compileInDir(runcmd, longdir, flags, importcfgfile, pkg.name, pkg.files...)
			if i == errPkg {
				if c.wantError && err == nil {
					return fmt.Errorf("compilation succeeded unexpectedly\n%s", out)
				} else if !c.wantError && err != nil {
					return err
				}
			} else if err != nil {
				return err
			}
			var fullshort []string
			for _, name := range pkg.files {
				fullshort = append(fullshort, filepath.Join(longdir, name), name)
			}
			if err := t.errorCheck(string(out), c.auto, fullshort...); err != nil {
				return err
			}
		}
		if c.action == "errorcheckdir" {
			return nil
		}
		fallthrough
	case "rundir":
		longdir := filepath.Join(t.gorootTestDir, t.goDirName())
		pkgs := goDirPackages(t.T, longdir, false)
		ldflags := []string{}
		flags := slices.Clone(buildFlags)
		for i, fl := range flags {
			if fl == "-ldflags" {
				ldflags = flags[i+1:]
				flags = flags[0:i]
				break
			}
		}
		importcfgfile := importcfg(pkgs)
		for i, pkg := range pkgs {
			_, err := compileInDir(runcmd, longdir, flags, importcfgfile, pkg.name, pkg.files...)
			if err != nil && !(c.wantError && c.action == "errorcheckandrundir" && i == len(pkgs)-2) {
				return err
			}
			if i == len(pkgs)-1 {
				if err := linkFile(runcmd, "a.exe", pkg.files[0], importcfgfile, ldflags); err != nil {
					return err
				}
				var cmd []string
				cmd = append(cmd, findExecCmd()...)
				cmd = append(cmd, filepath.Join(tempDir, "a.exe"))
				cmd = append(cmd, c.args...)
				out, err := runcmd(cmd...)
				if err != nil {
					return err
				}
				return t.checkExpectedOutput(out)
			}
		}
		return nil
	case "runindir":
		srcDir := filepath.Join(t.gorootTestDir, t.goDirName())
		modName := filepath.Base(srcDir)
		gopathSrcDir := filepath.Join(tempDir, "src", modName)
		if err := overlayDir(gopathSrcDir, srcDir); err != nil {
			t.Fatal(err)
		}
		modFile := fmt.Sprintf("module %s\ngo 1.14\n", modName)
		if err := os.WriteFile(filepath.Join(gopathSrcDir, "go.mod"), []byte(modFile), 0666); err != nil {
			t.Fatal(err)
		}
		indirCmd := func(args ...string) ([]byte, error) {
			cmd := exec.Command(args[0], args[1:]...)
			var buf bytes.Buffer
			cmd.Stdout = &buf
			cmd.Stderr = &buf
			cmd.Env = append(os.Environ(), "GOENV=off", "GOFLAGS=")
			cmd.Dir = gopathSrcDir
			cmd.Env = append(cmd.Env, "PWD="+gopathSrcDir)
			cmd.Env = append(cmd.Env, "GOPATH="+tempDir)
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
		cmd := []string{goTool, "run", t.goGcflags()}
		cmd = withLinkshared(cmd)
		cmd = append(cmd, buildFlags...)
		cmd = append(cmd, ".")
		out, err := indirCmd(cmd...)
		if err != nil {
			return err
		}
		return t.checkExpectedOutput(out)
	case "builddir", "buildrundir":
		longdir := filepath.Join(t.gorootTestDir, t.goDirName())
		files, err := os.ReadDir(longdir)
		if err != nil {
			t.Fatal(err)
		}
		var gos []string
		var asms []string
		for _, file := range files {
			switch filepath.Ext(file.Name()) {
			case ".go":
				gos = append(gos, filepath.Join(longdir, file.Name()))
			case ".s":
				asms = append(asms, filepath.Join(longdir, file.Name()))
			}
		}
		if len(asms) > 0 {
			emptyHdrFile := filepath.Join(tempDir, "go_asm.h")
			if err := os.WriteFile(emptyHdrFile, nil, 0666); err != nil {
				t.Fatalf("write empty go_asm.h: %v", err)
			}
			cmd := []string{goTool, "tool", "asm", "-p=main", "-gensymabis", "-o", "symabis"}
			cmd = append(cmd, asms...)
			if _, err := runcmd(cmd...); err != nil {
				return err
			}
		}
		var objs []string
		cmd := []string{goTool, "tool", "compile", "-p=main", "-e", "-D", ".", "-importcfg=" + stdlibImportcfgFile(), "-o", "go.o"}
		if len(asms) > 0 {
			cmd = append(cmd, "-asmhdr", "go_asm.h", "-symabis", "symabis")
		}
		cmd = append(cmd, gos...)
		if _, err := runcmd(cmd...); err != nil {
			return err
		}
		objs = append(objs, "go.o")
		if len(asms) > 0 {
			cmd := []string{goTool, "tool", "asm", "-p=main", "-e", "-I", ".", "-o", "asm.o"}
			cmd = append(cmd, asms...)
			if _, err := runcmd(cmd...); err != nil {
				return err
			}
			objs = append(objs, "asm.o")
		}
		cmd = []string{goTool, "tool", "pack", "c", "all.a"}
		cmd = append(cmd, objs...)
		if _, err := runcmd(cmd...); err != nil {
			return err
		}
		if err := linkFile(runcmd, "a.exe", "all.a", stdlibImportcfgFile(), nil); err != nil {
			return err
		}
		if c.action == "builddir" {
			return nil
		}
		cmd = append(findExecCmd(), filepath.Join(tempDir, "a.exe"))
		out, err := runcmd(cmd...)
		if err != nil {
			return err
		}
		return t.checkExpectedOutput(out)
	default:
		t.Fatalf("unimplemented action %q", c.action)
		panic("unreachable")
	}
}

func TestRouteLogosHarnessScopesToLogos(t *testing.T) {
	if routeLogosHarness("fixedbugs", "bug000.go", "run", []string{"-O=3"}) {
		t.Fatal("routeLogosHarness routed an upstream -O= file to the logos harness")
	}
	if !routeLogosHarness("logos", "run/twin_go.go", "run", nil) {
		t.Fatal("routeLogosHarness did not route a plain test/logos .go twin")
	}
	if !routeLogosHarness("logos", "run/x.go", "run", []string{"-O=3"}) {
		t.Fatal("routeLogosHarness did not route a -O= twin")
	}
	if !routeLogosHarness("logos", "run/hello.lgo", "run", nil) {
		t.Fatal("routeLogosHarness did not route a .lgo file")
	}
	if !routeLogosHarness("logos", "codegen/asm_twin.go", "asmcheck", nil) {
		t.Fatal("routeLogosHarness did not route a test/logos asmcheck twin")
	}
	if !routeLogosHarness("logos", "run/x.lgo", "rundir", nil) {
		t.Fatal("routeLogosHarness did not route a .lgo rundir case")
	}
	if routeLogosHarness("logos", "run/x.go", "skip", nil) {
		t.Fatal("routeLogosHarness routed a skip case to the logos harness")
	}
	if !routeLogosHarness("logos", "run/x.lgo", "compile", nil) {
		t.Fatal("routeLogosHarness did not route a .lgo compile action")
	}
}

func TestPinTargetFlagsMovesGOAMD64(t *testing.T) {
	c := pinTargetFlags(logosCase{
		flags:  []string{"-O=2", "-goamd64=v1", "-d=x"},
		runenv: []string{"GODEBUG=a=b"},
	})
	if got, want := strings.Join(c.flags, " "), "-O=2 -d=x"; got != want {
		t.Errorf("flags = %q, want %q", got, want)
	}
	if got, want := strings.Join(c.runenv, " "), "GODEBUG=a=b GOAMD64=v1"; got != want {
		t.Errorf("runenv = %q, want %q", got, want)
	}
}

func TestLogosGoBuildPreservesFlags(t *testing.T) {
	cmd := logosGoBuildCmd("-gcflags=all=", "x.go", []string{"-gcflags=-l"}, []string{"-O=3"})
	foundO, foundGC, fileLast := false, false, false
	for i, a := range cmd {
		if a == "-O=3" {
			foundO = true
		}
		if a == "-gcflags=-l" {
			foundGC = true
		}
		if a == "x.go" && i == len(cmd)-1 {
			fileLast = true
		}
	}
	if !foundO || !foundGC || !fileLast {
		t.Fatalf("logosGoBuildCmd dropped flags: %q", cmd)
	}
}

func TestLogosCompilerFlagsHasE(t *testing.T) {
	got := logosTestFlags([]string{"-m"})
	if !strings.HasPrefix(got, "-gcflags=") || !strings.Contains(got, "-e") || !strings.Contains(got, "-m") {
		t.Fatalf("logosTestFlags = %q, want -gcflags with -e and -m", got)
	}
}

func TestLogosAsmcheckEnvCarriesRunenv(t *testing.T) {
	t.Setenv("GOEXPERIMENT", "regabi")
	env := logosAsmcheckEnv(logosCase{runenv: []string{"GOEXPERIMENT=simd"}}, "linux/arm64/v8.1", nil)
	last := map[string]string{}
	for _, kv := range env {
		if k, v, ok := strings.Cut(kv, "="); ok {
			last[k] = v
		}
	}
	for k, want := range map[string]string{"GOOS": "linux", "GOARCH": "arm64", "GOARM64": "v8.1", "GOEXPERIMENT": "simd"} {
		if last[k] != want {
			t.Fatalf("logosAsmcheckEnv last %s = %q, want %q (env %q)", k, last[k], want, env)
		}
	}
	race := logosAsmcheckEnv(logosCase{}, "linux/amd64/", []string{"-race"})
	if !slices.Contains(race, "CGO_ENABLED=1") {
		t.Fatalf("race build lost CGO_ENABLED=1: %q", race)
	}
}
