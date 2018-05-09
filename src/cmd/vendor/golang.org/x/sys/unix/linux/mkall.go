// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// linux/mkall.go - Generates all Linux zsysnum, zsyscall, zerror, and ztype
// files for all 11 linux architectures supported by the go compiler. See
// README.md for more information about the build system.

// To run it you must have a git checkout of the Linux kernel and glibc. Once
// the appropriate sources are ready, the program is run as:
//     go run linux/mkall.go <linux_dir> <glibc_dir>

// +build ignore

package main

import (
	"bufio"
	"bytes"
	"debug/elf"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"unicode"
)

// These will be paths to the appropriate source directories.
var LinuxDir string
var GlibcDir string

const TempDir = "/tmp"
const IncludeDir = TempDir + "/include" // To hold our C headers
const BuildDir = TempDir + "/build"     // To hold intermediate build files

const GOOS = "linux"       // Only for Linux targets
const BuildArch = "amd64"  // Must be built on this architecture
const MinKernel = "2.6.23" // https://golang.org/doc/install#requirements

type target struct {
	GoArch     string // Architecture name according to Go
	LinuxArch  string // Architecture name according to the Linux Kernel
	GNUArch    string // Architecture name according to GNU tools (https://wiki.debian.org/Multiarch/Tuples)
	BigEndian  bool   // Default Little Endian
	SignedChar bool   // Is -fsigned-char needed (default no)
	Bits       int
}

// List of the 11 Linux targets supported by the go compiler. sparc64 is not
// currently supported, though a port is in progress.
var targets = []target{
	{
		GoArch:    "386",
		LinuxArch: "x86",
		GNUArch:   "i686-linux-gnu", // Note "i686" not "i386"
		Bits:      32,
	},
	{
		GoArch:    "amd64",
		LinuxArch: "x86",
		GNUArch:   "x86_64-linux-gnu",
		Bits:      64,
	},
	{
		GoArch:     "arm64",
		LinuxArch:  "arm64",
		GNUArch:    "aarch64-linux-gnu",
		SignedChar: true,
		Bits:       64,
	},
	{
		GoArch:    "arm",
		LinuxArch: "arm",
		GNUArch:   "arm-linux-gnueabi",
		Bits:      32,
	},
	{
		GoArch:    "mips",
		LinuxArch: "mips",
		GNUArch:   "mips-linux-gnu",
		BigEndian: true,
		Bits:      32,
	},
	{
		GoArch:    "mipsle",
		LinuxArch: "mips",
		GNUArch:   "mipsel-linux-gnu",
		Bits:      32,
	},
	{
		GoArch:    "mips64",
		LinuxArch: "mips",
		GNUArch:   "mips64-linux-gnuabi64",
		BigEndian: true,
		Bits:      64,
	},
	{
		GoArch:    "mips64le",
		LinuxArch: "mips",
		GNUArch:   "mips64el-linux-gnuabi64",
		Bits:      64,
	},
	{
		GoArch:    "ppc64",
		LinuxArch: "powerpc",
		GNUArch:   "powerpc64-linux-gnu",
		BigEndian: true,
		Bits:      64,
	},
	{
		GoArch:    "ppc64le",
		LinuxArch: "powerpc",
		GNUArch:   "powerpc64le-linux-gnu",
		Bits:      64,
	},
	{
		GoArch:     "s390x",
		LinuxArch:  "s390",
		GNUArch:    "s390x-linux-gnu",
		BigEndian:  true,
		SignedChar: true,
		Bits:       64,
	},
	// {
	// 	GoArch:    "sparc64",
	// 	LinuxArch: "sparc",
	// 	GNUArch:   "sparc64-linux-gnu",
	// 	BigEndian: true,
	// 	Bits:      64,
	// },
}

// ptracePairs is a list of pairs of targets that can, in some cases,
// run each other's binaries.
var ptracePairs = []struct{ a1, a2 string }{
	{"386", "amd64"},
	{"arm", "arm64"},
	{"mips", "mips64"},
	{"mipsle", "mips64le"},
}

func main() {
	if runtime.GOOS != GOOS || runtime.GOARCH != BuildArch {
		fmt.Printf("Build system has GOOS_GOARCH = %s_%s, need %s_%s\n",
			runtime.GOOS, runtime.GOARCH, GOOS, BuildArch)
		return
	}

	// Check that we are using the new build system if we should
	if os.Getenv("GOLANG_SYS_BUILD") != "docker" {
		fmt.Println("In the new build system, mkall.go should not be called directly.")
		fmt.Println("See README.md")
		return
	}

	// Parse the command line options
	if len(os.Args) != 3 {
		fmt.Println("USAGE: go run linux/mkall.go <linux_dir> <glibc_dir>")
		return
	}
	LinuxDir = os.Args[1]
	GlibcDir = os.Args[2]

	for _, t := range targets {
		fmt.Printf("----- GENERATING: %s -----\n", t.GoArch)
		if err := t.generateFiles(); err != nil {
			fmt.Printf("%v\n***** FAILURE:    %s *****\n\n", err, t.GoArch)
		} else {
			fmt.Printf("----- SUCCESS:    %s -----\n\n", t.GoArch)
		}
	}

	fmt.Printf("----- GENERATING ptrace pairs -----\n")
	ok := true
	for _, p := range ptracePairs {
		if err := generatePtracePair(p.a1, p.a2); err != nil {
			fmt.Printf("%v\n***** FAILURE: %s/%s *****\n\n", err, p.a1, p.a2)
			ok = false
		}
	}
	if ok {
		fmt.Printf("----- SUCCESS ptrace pairs    -----\n\n")
	}
}

// Makes an exec.Cmd with Stderr attached to os.Stderr
func makeCommand(name string, args ...string) *exec.Cmd {
	cmd := exec.Command(name, args...)
	cmd.Stderr = os.Stderr
	return cmd
}

// Runs the command, pipes output to a formatter, pipes that to an output file.
func (t *target) commandFormatOutput(formatter string, outputFile string,
	name string, args ...string) (err error) {
	mainCmd := makeCommand(name, args...)

	fmtCmd := makeCommand(formatter)
	if formatter == "mkpost" {
		fmtCmd = makeCommand("go", "run", "mkpost.go")
		// Set GOARCH_TARGET so mkpost knows what GOARCH is..
		fmtCmd.Env = append(os.Environ(), "GOARCH_TARGET="+t.GoArch)
		// Set GOARCH to host arch for mkpost, so it can run natively.
		for i, s := range fmtCmd.Env {
			if strings.HasPrefix(s, "GOARCH=") {
				fmtCmd.Env[i] = "GOARCH=" + BuildArch
			}
		}
	}

	// mainCmd | fmtCmd > outputFile
	if fmtCmd.Stdin, err = mainCmd.StdoutPipe(); err != nil {
		return
	}
	if fmtCmd.Stdout, err = os.Create(outputFile); err != nil {
		return
	}

	// Make sure the formatter eventually closes
	if err = fmtCmd.Start(); err != nil {
		return
	}
	defer func() {
		fmtErr := fmtCmd.Wait()
		if err == nil {
			err = fmtErr
		}
	}()

	return mainCmd.Run()
}

// Generates all the files for a Linux target
func (t *target) generateFiles() error {
	// Setup environment variables
	os.Setenv("GOOS", GOOS)
	os.Setenv("GOARCH", t.GoArch)

	// Get appropriate compiler and emulator (unless on x86)
	if t.LinuxArch != "x86" {
		// Check/Setup cross compiler
		compiler := t.GNUArch + "-gcc"
		if _, err := exec.LookPath(compiler); err != nil {
			return err
		}
		os.Setenv("CC", compiler)

		// Check/Setup emulator (usually first component of GNUArch)
		qemuArchName := t.GNUArch[:strings.Index(t.GNUArch, "-")]
		if t.LinuxArch == "powerpc" {
			qemuArchName = t.GoArch
		}
		os.Setenv("GORUN", "qemu-"+qemuArchName)
	} else {
		os.Setenv("CC", "gcc")
	}

	// Make the include directory and fill it with headers
	if err := os.MkdirAll(IncludeDir, os.ModePerm); err != nil {
		return err
	}
	defer os.RemoveAll(IncludeDir)
	if err := t.makeHeaders(); err != nil {
		return fmt.Errorf("could not make header files: %v", err)
	}
	fmt.Println("header files generated")

	// Make each of the four files
	if err := t.makeZSysnumFile(); err != nil {
		return fmt.Errorf("could not make zsysnum file: %v", err)
	}
	fmt.Println("zsysnum file generated")

	if err := t.makeZSyscallFile(); err != nil {
		return fmt.Errorf("could not make zsyscall file: %v", err)
	}
	fmt.Println("zsyscall file generated")

	if err := t.makeZTypesFile(); err != nil {
		return fmt.Errorf("could not make ztypes file: %v", err)
	}
	fmt.Println("ztypes file generated")

	if err := t.makeZErrorsFile(); err != nil {
		return fmt.Errorf("could not make zerrors file: %v", err)
	}
	fmt.Println("zerrors file generated")

	return nil
}

// Create the Linux, glibc and ABI (C compiler convention) headers in the include directory.
func (t *target) makeHeaders() error {
	// Make the Linux headers we need for this architecture
	linuxMake := makeCommand("make", "headers_install", "ARCH="+t.LinuxArch, "INSTALL_HDR_PATH="+TempDir)
	linuxMake.Dir = LinuxDir
	if err := linuxMake.Run(); err != nil {
		return err
	}

	// A Temporary build directory for glibc
	if err := os.MkdirAll(BuildDir, os.ModePerm); err != nil {
		return err
	}
	defer os.RemoveAll(BuildDir)

	// Make the glibc headers we need for this architecture
	confScript := filepath.Join(GlibcDir, "configure")
	glibcConf := makeCommand(confScript, "--prefix="+TempDir, "--host="+t.GNUArch, "--enable-kernel="+MinKernel)
	glibcConf.Dir = BuildDir
	if err := glibcConf.Run(); err != nil {
		return err
	}
	glibcMake := makeCommand("make", "install-headers")
	glibcMake.Dir = BuildDir
	if err := glibcMake.Run(); err != nil {
		return err
	}
	// We only need an empty stubs file
	stubsFile := filepath.Join(IncludeDir, "gnu/stubs.h")
	if file, err := os.Create(stubsFile); err != nil {
		return err
	} else {
		file.Close()
	}

	// ABI headers will specify C compiler behavior for the target platform.
	return t.makeABIHeaders()
}

// makeABIHeaders generates C header files based on the platform's calling convention.
// While many platforms have formal Application Binary Interfaces, in practice, whatever the
// dominant C compilers generate is the de-facto calling convention.
//
// We generate C headers instead of a Go file, so as to enable references to the ABI from Cgo.
func (t *target) makeABIHeaders() (err error) {
	abiDir := filepath.Join(IncludeDir, "abi")
	if err = os.Mkdir(abiDir, os.ModePerm); err != nil {
		return err
	}

	cc := os.Getenv("CC")
	if cc == "" {
		return errors.New("CC (compiler) env var not set")
	}

	// Build a sacrificial ELF file, to mine for C compiler behavior.
	binPath := filepath.Join(TempDir, "tmp_abi.o")
	bin, err := t.buildELF(cc, cCode, binPath)
	if err != nil {
		return fmt.Errorf("cannot build ELF to analyze: %v", err)
	}
	defer bin.Close()
	defer os.Remove(binPath)

	// Right now, we put everything in abi.h, but we may change this later.
	abiFile, err := os.Create(filepath.Join(abiDir, "abi.h"))
	if err != nil {
		return err
	}
	defer func() {
		if cerr := abiFile.Close(); cerr != nil && err == nil {
			err = cerr
		}
	}()

	if err = t.writeBitFieldMasks(bin, abiFile); err != nil {
		return fmt.Errorf("cannot write bitfield masks: %v", err)
	}

	return nil
}

func (t *target) buildELF(cc, src, path string) (*elf.File, error) {
	// Compile the cCode source using the set compiler - we will need its .data section.
	// Do not link the binary, so that we can find .data section offsets from the symbol values.
	ccCmd := makeCommand(cc, "-o", path, "-gdwarf", "-x", "c", "-c", "-")
	ccCmd.Stdin = strings.NewReader(src)
	ccCmd.Stdout = os.Stdout
	if err := ccCmd.Run(); err != nil {
		return nil, fmt.Errorf("compiler error: %v", err)
	}

	bin, err := elf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("cannot read ELF file %s: %v", path, err)
	}

	return bin, nil
}

func (t *target) writeBitFieldMasks(bin *elf.File, out io.Writer) error {
	symbols, err := bin.Symbols()
	if err != nil {
		return fmt.Errorf("getting ELF symbols: %v", err)
	}
	var masksSym *elf.Symbol

	for _, sym := range symbols {
		if sym.Name == "masks" {
			masksSym = &sym
		}
	}

	if masksSym == nil {
		return errors.New("could not find the 'masks' symbol in ELF symtab")
	}

	dataSection := bin.Section(".data")
	if dataSection == nil {
		return errors.New("ELF file has no .data section")
	}

	data, err := dataSection.Data()
	if err != nil {
		return fmt.Errorf("could not read .data section: %v\n", err)
	}

	var bo binary.ByteOrder
	if t.BigEndian {
		bo = binary.BigEndian
	} else {
		bo = binary.LittleEndian
	}

	// 64 bit masks of type uint64 are stored in the data section starting at masks.Value.
	// Here we are running on AMD64, but these values may be big endian or little endian,
	// depending on target architecture.
	for i := uint64(0); i < 64; i++ {
		off := masksSym.Value + i*8
		// Define each mask in native by order, so as to match target endian.
		fmt.Fprintf(out, "#define BITFIELD_MASK_%d %dULL\n", i, bo.Uint64(data[off:off+8]))
	}

	return nil
}

// makes the zsysnum_linux_$GOARCH.go file
func (t *target) makeZSysnumFile() error {
	zsysnumFile := fmt.Sprintf("zsysnum_linux_%s.go", t.GoArch)
	unistdFile := filepath.Join(IncludeDir, "asm/unistd.h")

	args := append(t.cFlags(), unistdFile)
	return t.commandFormatOutput("gofmt", zsysnumFile, "linux/mksysnum.pl", args...)
}

// makes the zsyscall_linux_$GOARCH.go file
func (t *target) makeZSyscallFile() error {
	zsyscallFile := fmt.Sprintf("zsyscall_linux_%s.go", t.GoArch)
	// Find the correct architecture syscall file (might end with x.go)
	archSyscallFile := fmt.Sprintf("syscall_linux_%s.go", t.GoArch)
	if _, err := os.Stat(archSyscallFile); os.IsNotExist(err) {
		shortArch := strings.TrimSuffix(t.GoArch, "le")
		archSyscallFile = fmt.Sprintf("syscall_linux_%sx.go", shortArch)
	}

	args := append(t.mksyscallFlags(), "-tags", "linux,"+t.GoArch,
		"syscall_linux.go", archSyscallFile)
	return t.commandFormatOutput("gofmt", zsyscallFile, "./mksyscall.pl", args...)
}

// makes the zerrors_linux_$GOARCH.go file
func (t *target) makeZErrorsFile() error {
	zerrorsFile := fmt.Sprintf("zerrors_linux_%s.go", t.GoArch)

	return t.commandFormatOutput("gofmt", zerrorsFile, "./mkerrors.sh", t.cFlags()...)
}

// makes the ztypes_linux_$GOARCH.go file
func (t *target) makeZTypesFile() error {
	ztypesFile := fmt.Sprintf("ztypes_linux_%s.go", t.GoArch)

	args := []string{"tool", "cgo", "-godefs", "--"}
	args = append(args, t.cFlags()...)
	args = append(args, "linux/types.go")
	return t.commandFormatOutput("mkpost", ztypesFile, "go", args...)
}

// Flags that should be given to gcc and cgo for this target
func (t *target) cFlags() []string {
	// Compile statically to avoid cross-architecture dynamic linking.
	flags := []string{"-Wall", "-Werror", "-static", "-I" + IncludeDir}

	// Architecture-specific flags
	if t.SignedChar {
		flags = append(flags, "-fsigned-char")
	}
	if t.LinuxArch == "x86" {
		flags = append(flags, fmt.Sprintf("-m%d", t.Bits))
	}

	return flags
}

// Flags that should be given to mksyscall for this target
func (t *target) mksyscallFlags() (flags []string) {
	if t.Bits == 32 {
		if t.BigEndian {
			flags = append(flags, "-b32")
		} else {
			flags = append(flags, "-l32")
		}
	}

	// This flag menas a 64-bit value should use (even, odd)-pair.
	if t.GoArch == "arm" || (t.LinuxArch == "mips" && t.Bits == 32) {
		flags = append(flags, "-arm")
	}
	return
}

// generatePtracePair takes a pair of GOARCH values that can run each
// other's binaries, such as 386 and amd64. It extracts the PtraceRegs
// type for each one. It writes a new file defining the types
// PtraceRegsArch1 and PtraceRegsArch2 and the corresponding functions
// Ptrace{Get,Set}Regs{arch1,arch2}. This permits debugging the other
// binary on a native system.
func generatePtracePair(arch1, arch2 string) error {
	def1, err := ptraceDef(arch1)
	if err != nil {
		return err
	}
	def2, err := ptraceDef(arch2)
	if err != nil {
		return err
	}
	f, err := os.Create(fmt.Sprintf("zptrace%s_linux.go", arch1))
	if err != nil {
		return err
	}
	buf := bufio.NewWriter(f)
	fmt.Fprintf(buf, "// Code generated by linux/mkall.go generatePtracePair(%s, %s). DO NOT EDIT.\n", arch1, arch2)
	fmt.Fprintf(buf, "\n")
	fmt.Fprintf(buf, "// +build linux\n")
	fmt.Fprintf(buf, "// +build %s %s\n", arch1, arch2)
	fmt.Fprintf(buf, "\n")
	fmt.Fprintf(buf, "package unix\n")
	fmt.Fprintf(buf, "\n")
	fmt.Fprintf(buf, "%s\n", `import "unsafe"`)
	fmt.Fprintf(buf, "\n")
	writeOnePtrace(buf, arch1, def1)
	fmt.Fprintf(buf, "\n")
	writeOnePtrace(buf, arch2, def2)
	if err := buf.Flush(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return nil
}

// ptraceDef returns the definition of PtraceRegs for arch.
func ptraceDef(arch string) (string, error) {
	filename := fmt.Sprintf("ztypes_linux_%s.go", arch)
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return "", fmt.Errorf("reading %s: %v", filename, err)
	}
	start := bytes.Index(data, []byte("type PtraceRegs struct"))
	if start < 0 {
		return "", fmt.Errorf("%s: no definition of PtraceRegs", filename)
	}
	data = data[start:]
	end := bytes.Index(data, []byte("\n}\n"))
	if end < 0 {
		return "", fmt.Errorf("%s: can't find end of PtraceRegs definition", filename)
	}
	return string(data[:end+2]), nil
}

// writeOnePtrace writes out the ptrace definitions for arch.
func writeOnePtrace(w io.Writer, arch, def string) {
	uarch := string(unicode.ToUpper(rune(arch[0]))) + arch[1:]
	fmt.Fprintf(w, "// PtraceRegs%s is the registers used by %s binaries.\n", uarch, arch)
	fmt.Fprintf(w, "%s\n", strings.Replace(def, "PtraceRegs", "PtraceRegs"+uarch, 1))
	fmt.Fprintf(w, "\n")
	fmt.Fprintf(w, "// PtraceGetRegs%s fetches the registers used by %s binaries.\n", uarch, arch)
	fmt.Fprintf(w, "func PtraceGetRegs%s(pid int, regsout *PtraceRegs%s) error {\n", uarch, uarch)
	fmt.Fprintf(w, "\treturn ptrace(PTRACE_GETREGS, pid, 0, uintptr(unsafe.Pointer(regsout)))\n")
	fmt.Fprintf(w, "}\n")
	fmt.Fprintf(w, "\n")
	fmt.Fprintf(w, "// PtraceSetRegs%s sets the registers used by %s binaries.\n", uarch, arch)
	fmt.Fprintf(w, "func PtraceSetRegs%s(pid int, regs *PtraceRegs%s) error {\n", uarch, uarch)
	fmt.Fprintf(w, "\treturn ptrace(PTRACE_SETREGS, pid, 0, uintptr(unsafe.Pointer(regs)))\n")
	fmt.Fprintf(w, "}\n")
}

// cCode is compiled for the target architecture, and the resulting data section is carved for
// the statically initialized bit masks.
const cCode = `
// Bit fields are used in some system calls and other ABIs, but their memory layout is
// implementation-defined [1]. Even with formal ABIs, bit fields are a source of subtle bugs [2].
// Here we generate the offsets for all 64 bits in an uint64.
// 1: http://en.cppreference.com/w/c/language/bit_field
// 2: https://lwn.net/Articles/478657/

#include <stdint.h>

struct bitfield {
	union {
		uint64_t val;
		struct {
			uint64_t u64_bit_0 : 1;
			uint64_t u64_bit_1 : 1;
			uint64_t u64_bit_2 : 1;
			uint64_t u64_bit_3 : 1;
			uint64_t u64_bit_4 : 1;
			uint64_t u64_bit_5 : 1;
			uint64_t u64_bit_6 : 1;
			uint64_t u64_bit_7 : 1;
			uint64_t u64_bit_8 : 1;
			uint64_t u64_bit_9 : 1;
			uint64_t u64_bit_10 : 1;
			uint64_t u64_bit_11 : 1;
			uint64_t u64_bit_12 : 1;
			uint64_t u64_bit_13 : 1;
			uint64_t u64_bit_14 : 1;
			uint64_t u64_bit_15 : 1;
			uint64_t u64_bit_16 : 1;
			uint64_t u64_bit_17 : 1;
			uint64_t u64_bit_18 : 1;
			uint64_t u64_bit_19 : 1;
			uint64_t u64_bit_20 : 1;
			uint64_t u64_bit_21 : 1;
			uint64_t u64_bit_22 : 1;
			uint64_t u64_bit_23 : 1;
			uint64_t u64_bit_24 : 1;
			uint64_t u64_bit_25 : 1;
			uint64_t u64_bit_26 : 1;
			uint64_t u64_bit_27 : 1;
			uint64_t u64_bit_28 : 1;
			uint64_t u64_bit_29 : 1;
			uint64_t u64_bit_30 : 1;
			uint64_t u64_bit_31 : 1;
			uint64_t u64_bit_32 : 1;
			uint64_t u64_bit_33 : 1;
			uint64_t u64_bit_34 : 1;
			uint64_t u64_bit_35 : 1;
			uint64_t u64_bit_36 : 1;
			uint64_t u64_bit_37 : 1;
			uint64_t u64_bit_38 : 1;
			uint64_t u64_bit_39 : 1;
			uint64_t u64_bit_40 : 1;
			uint64_t u64_bit_41 : 1;
			uint64_t u64_bit_42 : 1;
			uint64_t u64_bit_43 : 1;
			uint64_t u64_bit_44 : 1;
			uint64_t u64_bit_45 : 1;
			uint64_t u64_bit_46 : 1;
			uint64_t u64_bit_47 : 1;
			uint64_t u64_bit_48 : 1;
			uint64_t u64_bit_49 : 1;
			uint64_t u64_bit_50 : 1;
			uint64_t u64_bit_51 : 1;
			uint64_t u64_bit_52 : 1;
			uint64_t u64_bit_53 : 1;
			uint64_t u64_bit_54 : 1;
			uint64_t u64_bit_55 : 1;
			uint64_t u64_bit_56 : 1;
			uint64_t u64_bit_57 : 1;
			uint64_t u64_bit_58 : 1;
			uint64_t u64_bit_59 : 1;
			uint64_t u64_bit_60 : 1;
			uint64_t u64_bit_61 : 1;
			uint64_t u64_bit_62 : 1;
			uint64_t u64_bit_63 : 1;
		};
	};
};

struct bitfield masks[] = {
	{.u64_bit_0 = 1},
	{.u64_bit_1 = 1},
	{.u64_bit_2 = 1},
	{.u64_bit_3 = 1},
	{.u64_bit_4 = 1},
	{.u64_bit_5 = 1},
	{.u64_bit_6 = 1},
	{.u64_bit_7 = 1},
	{.u64_bit_8 = 1},
	{.u64_bit_9 = 1},
	{.u64_bit_10 = 1},
	{.u64_bit_11 = 1},
	{.u64_bit_12 = 1},
	{.u64_bit_13 = 1},
	{.u64_bit_14 = 1},
	{.u64_bit_15 = 1},
	{.u64_bit_16 = 1},
	{.u64_bit_17 = 1},
	{.u64_bit_18 = 1},
	{.u64_bit_19 = 1},
	{.u64_bit_20 = 1},
	{.u64_bit_21 = 1},
	{.u64_bit_22 = 1},
	{.u64_bit_23 = 1},
	{.u64_bit_24 = 1},
	{.u64_bit_25 = 1},
	{.u64_bit_26 = 1},
	{.u64_bit_27 = 1},
	{.u64_bit_28 = 1},
	{.u64_bit_29 = 1},
	{.u64_bit_30 = 1},
	{.u64_bit_31 = 1},
	{.u64_bit_32 = 1},
	{.u64_bit_33 = 1},
	{.u64_bit_34 = 1},
	{.u64_bit_35 = 1},
	{.u64_bit_36 = 1},
	{.u64_bit_37 = 1},
	{.u64_bit_38 = 1},
	{.u64_bit_39 = 1},
	{.u64_bit_40 = 1},
	{.u64_bit_41 = 1},
	{.u64_bit_42 = 1},
	{.u64_bit_43 = 1},
	{.u64_bit_44 = 1},
	{.u64_bit_45 = 1},
	{.u64_bit_46 = 1},
	{.u64_bit_47 = 1},
	{.u64_bit_48 = 1},
	{.u64_bit_49 = 1},
	{.u64_bit_50 = 1},
	{.u64_bit_51 = 1},
	{.u64_bit_52 = 1},
	{.u64_bit_53 = 1},
	{.u64_bit_54 = 1},
	{.u64_bit_55 = 1},
	{.u64_bit_56 = 1},
	{.u64_bit_57 = 1},
	{.u64_bit_58 = 1},
	{.u64_bit_59 = 1},
	{.u64_bit_60 = 1},
	{.u64_bit_61 = 1},
	{.u64_bit_62 = 1},
	{.u64_bit_63 = 1}
};

int main(int argc, char **argv) {
	struct bitfield *mask_ptr = &masks[0];
	return mask_ptr->val;
}

`
