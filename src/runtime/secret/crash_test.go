// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimesecret && linux

package secret

import (
	"bytes"
	"debug/elf"
	"fmt"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
)

// Copied from runtime/runtime-gdb_unix_test.go
func canGenerateCore(t *testing.T) bool {
	// Ensure there is enough RLIMIT_CORE available to generate a full core.
	var lim syscall.Rlimit
	err := syscall.Getrlimit(syscall.RLIMIT_CORE, &lim)
	if err != nil {
		t.Fatalf("error getting rlimit: %v", err)
	}
	// Minimum RLIMIT_CORE max to allow. This is a conservative estimate.
	// Most systems allow infinity.
	const minRlimitCore = 100 << 20 // 100 MB
	if lim.Max < minRlimitCore {
		t.Skipf("RLIMIT_CORE max too low: %#+v", lim)
	}

	// Make sure core pattern will send core to the current directory.
	b, err := os.ReadFile("/proc/sys/kernel/core_pattern")
	if err != nil {
		t.Fatalf("error reading core_pattern: %v", err)
	}
	if string(b) != "core\n" {
		t.Skipf("Unexpected core pattern %q", string(b))
	}

	coreUsesPID := false
	b, err = os.ReadFile("/proc/sys/kernel/core_uses_pid")
	if err == nil {
		switch string(bytes.TrimSpace(b)) {
		case "0":
		case "1":
			coreUsesPID = true
		default:
			t.Skipf("unexpected core_uses_pid value %q", string(b))
		}
	}
	return coreUsesPID
}

func TestCore(t *testing.T) {
	// use secret, grab a coredump, rummage through
	// it, trying to find our secret.

	switch runtime.GOARCH {
	case "amd64", "arm64":
	default:
		t.Skip("unsupported arch")
	}
	coreUsesPid := canGenerateCore(t)

	// Build our crashing program
	// Because we need assembly files to properly dirty our state
	// we need to construct a package in our temporary directory.
	tmpDir := t.TempDir()
	// copy our base source
	err := copyToDir("./testdata/crash.go", tmpDir, nil)
	if err != nil {
		t.Fatalf("error copying directory %v", err)
	}
	// Copy our testing assembly files. Use the ones from the package
	// to assure that they are always in sync
	err = copyToDir("./asm_amd64.s", tmpDir, nil)
	if err != nil {
		t.Fatalf("error copying file %v", err)
	}
	err = copyToDir("./asm_arm64.s", tmpDir, nil)
	if err != nil {
		t.Fatalf("error copying file %v", err)
	}
	err = copyToDir("./stubs.go", tmpDir, func(s string) string {
		return strings.Replace(s, "package secret", "package main", 1)
	})
	if err != nil {
		t.Fatalf("error copying file %v", err)
	}

	// the crashing package will live out of tree, so its source files
	// cannot refer to our internal packages. However, the assembly files
	// can refer to internal names and we can pass the missing offsets as
	// a small generated file
	offsets := `
	package main
	const (
		offsetX86HasAVX    = %v
		offsetX86HasAVX512 = %v
	)
	`
	err = os.WriteFile(filepath.Join(tmpDir, "offsets.go"), []byte(fmt.Sprintf(offsets, offsetX86HasAVX, offsetX86HasAVX512)), 0666)
	if err != nil {
		t.Fatalf("error writing offset file %v", err)
	}

	// generate go.mod file
	cmd := exec.Command(testenv.GoToolPath(t), "mod", "init", "crashtest")
	cmd.Dir = tmpDir
	out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("error initing module %v\n%s", err, out)
	}

	cmd = exec.Command(testenv.GoToolPath(t), "build", "-o", filepath.Join(tmpDir, "a.exe"))
	cmd.Dir = tmpDir
	out, err = testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("error building source %v\n%s", err, out)
	}

	// Start the test binary.
	cmd = testenv.CommandContext(t, t.Context(), "./a.exe")
	cmd.Dir = tmpDir
	var stdout strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stdout

	err = cmd.Run()
	// For debugging.
	t.Logf("\n\n\n--- START SUBPROCESS ---\n\n\n%s\n\n--- END SUBPROCESS ---\n\n\n", stdout.String())
	if err == nil {
		t.Fatalf("test binary did not crash")
	}
	eErr, ok := err.(*exec.ExitError)
	if !ok {
		t.Fatalf("error is not exit error: %v", err)
	}
	if eErr.Exited() {
		t.Fatalf("process exited instead of being terminated: %v", eErr)
	}

	rummage(t, tmpDir, eErr.Pid(), coreUsesPid)
}

func copyToDir(name string, dir string, replace func(string) string) error {
	f, err := os.ReadFile(name)
	if err != nil {
		return err
	}
	if replace != nil {
		f = []byte(replace(string(f)))
	}
	return os.WriteFile(filepath.Join(dir, filepath.Base(name)), f, 0666)
}

type violation struct {
	id  byte   // secret ID
	off uint64 // offset in core dump
}

// A secret value that should never appear in a core dump,
// except for this global variable itself.
// The first byte of the secret is variable, to track
// different instances of it.
//
// If this value is changed, update ./internal/crashsecret/main.go
// TODO: this is little-endian specific.
var secretStore = [8]byte{
	0x00,
	0x81,
	0xa0,
	0xc6,
	0xb3,
	0x01,
	0x66,
	0x53,
}

func rummage(t *testing.T, tmpDir string, pid int, coreUsesPid bool) {
	coreFileName := "core"
	if coreUsesPid {
		coreFileName += fmt.Sprintf(".%d", pid)
	}
	core, err := os.Open(filepath.Join(tmpDir, coreFileName))
	if err != nil {
		t.Fatalf("core file not found: %v", err)
	}
	b, err := io.ReadAll(core)
	if err != nil {
		t.Fatalf("can't read core file: %v", err)
	}

	// Open elf view onto core file.
	coreElf, err := elf.NewFile(core)
	if err != nil {
		t.Fatalf("can't parse core file: %v", err)
	}

	// Look for any places that have the secret.
	var violations []violation // core file offsets where we found a secret
	i := 0
	for {
		j := bytes.Index(b[i:], secretStore[1:])
		if j < 0 {
			break
		}
		j--
		i += j

		t.Errorf("secret %d found at offset %x in core file", b[i], i)
		violations = append(violations, violation{
			id:  b[i],
			off: uint64(i),
		})

		i += len(secretStore)
	}

	// Get more specific data about where in the core we found the secrets.
	regions := elfRegions(t, core, coreElf)
	for _, r := range regions {
		for _, v := range violations {
			if v.off >= r.min && v.off < r.max {
				var addr string
				if r.addrMin != 0 {
					addr = fmt.Sprintf(" addr=%x", r.addrMin+(v.off-r.min))
				}
				t.Logf("additional info: secret %d at offset %x in %s%s", v.id, v.off-r.min, r.name, addr)
			}
		}
	}
}

type elfRegion struct {
	name             string
	min, max         uint64 // core file offset range
	addrMin, addrMax uint64 // inferior address range (or 0,0 if no address, like registers)
}

func elfRegions(t *testing.T, core *os.File, coreElf *elf.File) []elfRegion {
	var regions []elfRegion
	for _, p := range coreElf.Progs {
		regions = append(regions, elfRegion{
			name:    fmt.Sprintf("%s[%s]", p.Type, p.Flags),
			min:     p.Off,
			max:     p.Off + min(p.Filesz, p.Memsz),
			addrMin: p.Vaddr,
			addrMax: p.Vaddr + min(p.Filesz, p.Memsz),
		})
	}

	// TODO(dmo): parse thread regions for arm64.
	// This doesn't invalidate the test, it just makes it harder to figure
	// out where we're leaking stuff.
	if runtime.GOARCH == "amd64" {
		regions = append(regions, threadRegions(t, core, coreElf)...)
	}

	for i, r1 := range regions {
		for j, r2 := range regions {
			if i == j {
				continue
			}
			if r1.max <= r2.min || r2.max <= r1.min {
				continue
			}
			t.Fatalf("overlapping regions %v %v", r1, r2)
		}
	}

	return regions
}

func threadRegions(t *testing.T, core *os.File, coreElf *elf.File) []elfRegion {
	var regions []elfRegion

	for _, prog := range coreElf.Progs {
		if prog.Type != elf.PT_NOTE {
			continue
		}

		b := make([]byte, prog.Filesz)
		_, err := core.ReadAt(b, int64(prog.Off))
		if err != nil {
			t.Fatalf("can't read core file %v", err)
		}
		prefix := "unk"
		b0 := b
		for len(b) > 0 {
			namesz := coreElf.ByteOrder.Uint32(b)
			b = b[4:]
			descsz := coreElf.ByteOrder.Uint32(b)
			b = b[4:]
			typ := elf.NType(coreElf.ByteOrder.Uint32(b))
			b = b[4:]
			name := string(b[:namesz-1])
			b = b[(namesz+3)/4*4:]
			off := prog.Off + uint64(len(b0)-len(b))
			desc := b[:descsz]
			b = b[(descsz+3)/4*4:]

			if name != "CORE" && name != "LINUX" {
				continue
			}
			end := off + uint64(len(desc))
			// Note: amd64 specific
			// See /usr/include/x86_64-linux-gnu/bits/sigcontext.h
			//
			//   struct _fpstate
			switch typ {
			case elf.NT_PRSTATUS:
				pid := coreElf.ByteOrder.Uint32(desc[32:36])
				prefix = fmt.Sprintf("thread%d: ", pid)
				regions = append(regions, elfRegion{
					name: prefix + "prstatus header",
					min:  off,
					max:  off + 112,
				})
				off += 112
				greg := []string{
					"r15",
					"r14",
					"r13",
					"r12",
					"rbp",
					"rbx",
					"r11",
					"r10",
					"r9",
					"r8",
					"rax",
					"rcx",
					"rdx",
					"rsi",
					"rdi",
					"orig_rax",
					"rip",
					"cs",
					"eflags",
					"rsp",
					"ss",
					"fs_base",
					"gs_base",
					"ds",
					"es",
					"fs",
					"gs",
				}
				for _, r := range greg {
					regions = append(regions, elfRegion{
						name: prefix + r,
						min:  off,
						max:  off + 8,
					})
					off += 8
				}
				regions = append(regions, elfRegion{
					name: prefix + "prstatus footer",
					min:  off,
					max:  off + 8,
				})
				off += 8
			case elf.NT_FPREGSET:
				regions = append(regions, elfRegion{
					name: prefix + "fpregset header",
					min:  off,
					max:  off + 32,
				})
				off += 32
				for i := 0; i < 8; i++ {
					regions = append(regions, elfRegion{
						name: prefix + fmt.Sprintf("mmx%d", i),
						min:  off,
						max:  off + 16,
					})
					off += 16
					// They are long double (10 bytes), but
					// stored in 16-byte slots.
				}
				for i := 0; i < 16; i++ {
					regions = append(regions, elfRegion{
						name: prefix + fmt.Sprintf("xmm%d", i),
						min:  off,
						max:  off + 16,
					})
					off += 16
				}
				regions = append(regions, elfRegion{
					name: prefix + "fpregset footer",
					min:  off,
					max:  off + 96,
				})
				off += 96
				/*
					case NT_X86_XSTATE: // aka NT_PRPSINFO+511
						// legacy: 512 bytes
						// xsave header: 64 bytes
						fmt.Printf("hdr %v\n", desc[512:][:64])
						// ymm high128: 256 bytes

						println(len(desc))
						fallthrough
				*/
			default:
				regions = append(regions, elfRegion{
					name: fmt.Sprintf("%s/%s", name, typ),
					min:  off,
					max:  off + uint64(len(desc)),
				})
				off += uint64(len(desc))
			}
			if off != end {
				t.Fatalf("note section incomplete")
			}
		}
	}
	return regions
}
