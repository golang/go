// run

package main

// M19-S6 attack (SPEC Sec 19.1, T-301.2): on amd64 the sequential scan of a
// Green Tea span uses hardware prefetch. The program builds a probe with the
// toolchain under test (the linked program keeps its symbols, the run action's
// own binary does not), disassembles it with the go tool from the same GOROOT,
// takes every PREFETCHT0 the span deck emitted (source file logos_spandeck.go)
// and requires each one to sit inside a loop, that is, to have a jump back to
// an address at or before it later in the same function. The deck holds three
// prefetch sites: the pointer slots of a grey object, the first window of
// pointer targets, and the target a window ahead of the one being marked.
// Other architectures and collectors without the span queue (no
// runtime.scanSpan) print nothing.

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

const (
	tool          = "go"
	probeFile     = "main.go"
	deckFile      = "logos_spandeck.go"
	deckSites     = 3
	symbolPattern = `^runtime\.(scanSpan|scanObjectSmall|scanObjectsSmall|logos)`
	probeSrc      = "package main\n\nfunc main() {}\n"
)

type insn struct {
	addr   uint64
	file   string
	op     string
	target uint64
}

func command(dir string, name string, args ...string) error {
	cmd := exec.Command(filepath.Join(runtime.GOROOT(), "bin", name), args...)
	cmd.Dir = dir
	// Span deck selection is compile time (GOEXPERIMENT=logosspandeck) and
	// the default build has it off, so enable it explicitly for the probe
	// build. GOEXPERIMENT parsing is in order and later entries win, so
	// appending keeps the inherited experiments and overrides any prior
	// nologosspandeck.
	exp := os.Getenv("GOEXPERIMENT")
	if exp != "" {
		exp += ","
	}
	cmd.Env = append(os.Environ(), "GOEXPERIMENT="+exp+"logosspandeck")
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("%s %s: %w\n%s", name, strings.Join(args, " "), err, out)
	}
	return nil
}

func disassemble(dir string) (string, error) {
	cmd := exec.Command(filepath.Join(runtime.GOROOT(), "bin", "go"), "tool", "objdump", "-s", symbolPattern, filepath.Join(dir, "probe.exe"))
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("objdump: %w\n%s", err, out)
	}
	return string(out), nil
}

func buildProbe(dir string) error {
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module probe\n\ngo 1.27\n"), 0644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, probeFile), []byte(probeSrc), 0644); err != nil {
		return err
	}
	return command(dir, tool, "build", "-o", "probe.exe", ".")
}

func parseInsn(line string) (insn, bool) {
	f := strings.Fields(line)
	if len(f) < 4 || !strings.HasPrefix(f[1], "0x") {
		return insn{}, false
	}
	addr, err := strconv.ParseUint(f[1], 0, 64)
	if err != nil {
		return insn{}, false
	}
	in := insn{addr: addr, file: strings.SplitN(f[0], ":", 2)[0], op: f[3]}
	if strings.HasPrefix(in.op, "J") && len(f) > 4 && strings.HasPrefix(f[4], "0x") {
		if in.target, err = strconv.ParseUint(f[4], 0, 64); err != nil {
			return insn{}, false
		}
	}
	return in, true
}

func functions(dump string) [][]insn {
	var fns [][]insn
	for _, line := range strings.Split(dump, "\n") {
		if strings.HasPrefix(line, "TEXT ") {
			fns = append(fns, nil)
			continue
		}
		if in, ok := parseInsn(line); ok && len(fns) > 0 {
			fns[len(fns)-1] = append(fns[len(fns)-1], in)
		}
	}
	return fns
}

func inLoop(fn []insn, pf insn) bool {
	for _, in := range fn {
		if in.addr > pf.addr && in.target != 0 && in.target <= pf.addr {
			return true
		}
	}
	return false
}

func countSites(dump string) (int, error) {
	sites := 0
	for _, fn := range functions(dump) {
		for _, in := range fn {
			if in.file != deckFile || in.op != "PREFETCHT0" {
				continue
			}
			sites++
			if !inLoop(fn, in) {
				return 0, fmt.Errorf("PREFETCHT0 at %#x in %s is not in a loop", in.addr, deckFile)
			}
		}
	}
	return sites, nil
}

func check() error {
	if runtime.GOARCH != "amd64" {
		return nil
	}
	dir, err := os.MkdirTemp("", "prefetch")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	if err := buildProbe(dir); err != nil {
		return err
	}
	dump, err := disassemble(dir)
	if err != nil {
		return err
	}
	if !strings.Contains(dump, "TEXT runtime.scanSpan(SB)") {
		return nil
	}
	sites, err := countSites(dump)
	if err != nil {
		return err
	}
	if sites < deckSites {
		return fmt.Errorf("%s holds %d PREFETCHT0, want %d", deckFile, sites, deckSites)
	}
	return nil
}

func main() {
	if err := check(); err != nil {
		fmt.Println("FAIL", err)
		os.Exit(1)
	}
}
