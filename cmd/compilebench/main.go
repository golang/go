// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Compilebench benchmarks the speed of the Go compiler.
//
// Usage:
//
//	compilebench [options]
//
// It times the compilation of various packages and prints results in
// the format used by package testing (and expected by golang.org/x/perf/cmd/benchstat).
//
// The options are:
//
//	-alloc
//		Report allocations.
//
//	-compile exe
//		Use exe as the path to the cmd/compile binary.
//
//	-compileflags 'list'
//		Pass the space-separated list of flags to the compilation.
//
//	-link exe
//		Use exe as the path to the cmd/link binary.
//
//	-linkflags 'list'
//		Pass the space-separated list of flags to the linker.
//
//	-count n
//		Run each benchmark n times (default 1).
//
//	-cpuprofile file
//		Write a CPU profile of the compiler to file.
//
//	-go path
//		Path to "go" command (default "go").
//
//	-memprofile file
//		Write a memory profile of the compiler to file.
//
//	-memprofilerate rate
//		Set runtime.MemProfileRate during compilation.
//
//	-obj
//		Report object file statistics.
//
//	-pkg pkg
//		Benchmark compiling a single package.
//
//	-run regexp
//		Only run benchmarks with names matching regexp.
//
//	-short
//		Skip long-running benchmarks.
//
// Although -cpuprofile and -memprofile are intended to write a
// combined profile for all the executed benchmarks to file,
// today they write only the profile for the last benchmark executed.
//
// The default memory profiling rate is one profile sample per 512 kB
// allocated (see “go doc runtime.MemProfileRate”).
// Lowering the rate (for example, -memprofilerate 64000) produces
// a more fine-grained and therefore accurate profile, but it also incurs
// execution cost. For benchmark comparisons, never use timings
// obtained with a low -memprofilerate option.
//
// # Example
//
// Assuming the base version of the compiler has been saved with
// “toolstash save,” this sequence compares the old and new compiler:
//
//	compilebench -count 10 -compile $(toolstash -n compile) >old.txt
//	compilebench -count 10 >new.txt
//	benchstat old.txt new.txt
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	exec "golang.org/x/sys/execabs"
)

var (
	goroot                   string
	compiler                 string
	assembler                string
	linker                   string
	runRE                    *regexp.Regexp
	is6g                     bool
	needCompilingRuntimeFlag bool
)

var (
	flagGoCmd          = flag.String("go", "go", "path to \"go\" command")
	flagAlloc          = flag.Bool("alloc", false, "report allocations")
	flagObj            = flag.Bool("obj", false, "report object file stats")
	flagCompiler       = flag.String("compile", "", "use `exe` as the cmd/compile binary")
	flagAssembler      = flag.String("asm", "", "use `exe` as the cmd/asm binary")
	flagCompilerFlags  = flag.String("compileflags", "", "additional `flags` to pass to compile")
	flagLinker         = flag.String("link", "", "use `exe` as the cmd/link binary")
	flagLinkerFlags    = flag.String("linkflags", "", "additional `flags` to pass to link")
	flagRun            = flag.String("run", "", "run benchmarks matching `regexp`")
	flagCount          = flag.Int("count", 1, "run benchmarks `n` times")
	flagCpuprofile     = flag.String("cpuprofile", "", "write CPU profile to `file`")
	flagMemprofile     = flag.String("memprofile", "", "write memory profile to `file`")
	flagMemprofilerate = flag.Int64("memprofilerate", -1, "set memory profile `rate`")
	flagPackage        = flag.String("pkg", "", "if set, benchmark the package at path `pkg`")
	flagShort          = flag.Bool("short", false, "skip long-running benchmarks")
	flagTrace          = flag.Bool("trace", false, "debug tracing of builds")
)

type test struct {
	name string
	r    runner
}

type runner interface {
	long() bool
	run(name string, count int) error
}

var tests = []test{
	{"BenchmarkTemplate", compile{"html/template"}},
	{"BenchmarkUnicode", compile{"unicode"}},
	{"BenchmarkGoTypes", compile{"go/types"}},
	{"BenchmarkCompiler", compile{"cmd/compile/internal/gc"}},
	{"BenchmarkSSA", compile{"cmd/compile/internal/ssa"}},
	{"BenchmarkFlate", compile{"compress/flate"}},
	{"BenchmarkGoParser", compile{"go/parser"}},
	{"BenchmarkReflect", compile{"reflect"}},
	{"BenchmarkTar", compile{"archive/tar"}},
	{"BenchmarkXML", compile{"encoding/xml"}},
	{"BenchmarkLinkCompiler", link{"cmd/compile", ""}},
	{"BenchmarkExternalLinkCompiler", link{"cmd/compile", "-linkmode=external"}},
	{"BenchmarkLinkWithoutDebugCompiler", link{"cmd/compile", "-w"}},
	{"BenchmarkStdCmd", goBuild{[]string{"std", "cmd"}}},
	{"BenchmarkHelloSize", size{"$GOROOT/test/helloworld.go", false}},
	{"BenchmarkCmdGoSize", size{"cmd/go", true}},
}

func usage() {
	fmt.Fprintf(os.Stderr, "usage: compilebench [options]\n")
	fmt.Fprintf(os.Stderr, "options:\n")
	flag.PrintDefaults()
	os.Exit(2)
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("compilebench: ")
	flag.Usage = usage
	flag.Parse()
	if flag.NArg() != 0 {
		usage()
	}

	s, err := exec.Command(*flagGoCmd, "env", "GOROOT").CombinedOutput()
	if err != nil {
		log.Fatalf("%s env GOROOT: %v", *flagGoCmd, err)
	}
	goroot = strings.TrimSpace(string(s))
	os.Setenv("GOROOT", goroot) // for any subcommands

	compiler = *flagCompiler
	if compiler == "" {
		var foundTool string
		foundTool, compiler = toolPath("compile", "6g")
		if foundTool == "6g" {
			is6g = true
		}
	}
	assembler = *flagAssembler
	if assembler == "" {
		_, assembler = toolPath("asm")
	}
	if err := checkCompilingRuntimeFlag(assembler); err != nil {
		log.Fatalf("checkCompilingRuntimeFlag: %v", err)
	}

	linker = *flagLinker
	if linker == "" && !is6g { // TODO: Support 6l
		_, linker = toolPath("link")
	}

	if is6g {
		*flagMemprofilerate = -1
		*flagAlloc = false
		*flagCpuprofile = ""
		*flagMemprofile = ""
	}

	if *flagRun != "" {
		r, err := regexp.Compile(*flagRun)
		if err != nil {
			log.Fatalf("invalid -run argument: %v", err)
		}
		runRE = r
	}

	if *flagPackage != "" {
		tests = []test{
			{"BenchmarkPkg", compile{*flagPackage}},
			{"BenchmarkPkgLink", link{*flagPackage, ""}},
		}
		runRE = nil
	}

	for i := 0; i < *flagCount; i++ {
		for _, tt := range tests {
			if tt.r.long() && *flagShort {
				continue
			}
			if runRE == nil || runRE.MatchString(tt.name) {
				if err := tt.r.run(tt.name, i); err != nil {
					log.Printf("%s: %v", tt.name, err)
				}
			}
		}
	}
}

func toolPath(names ...string) (found, path string) {
	var out1 []byte
	var err1 error
	for i, name := range names {
		out, err := exec.Command(*flagGoCmd, "tool", "-n", name).CombinedOutput()
		if err == nil {
			return name, strings.TrimSpace(string(out))
		}
		if i == 0 {
			out1, err1 = out, err
		}
	}
	log.Fatalf("go tool -n %s: %v\n%s", names[0], err1, out1)
	return "", ""
}

type Pkg struct {
	ImportPath string
	Dir        string
	GoFiles    []string
	SFiles     []string
}

func goList(dir string) (*Pkg, error) {
	var pkg Pkg
	out, err := exec.Command(*flagGoCmd, "list", "-json", dir).Output()
	if err != nil {
		return nil, fmt.Errorf("go list -json %s: %v", dir, err)
	}
	if err := json.Unmarshal(out, &pkg); err != nil {
		return nil, fmt.Errorf("go list -json %s: unmarshal: %v", dir, err)
	}
	return &pkg, nil
}

func runCmd(name string, cmd *exec.Cmd) error {
	start := time.Now()
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%v\n%s", err, out)
	}
	fmt.Printf("%s 1 %d ns/op\n", name, time.Since(start).Nanoseconds())
	return nil
}

type goBuild struct{ pkgs []string }

func (goBuild) long() bool { return true }

func (r goBuild) run(name string, count int) error {
	args := []string{"build", "-a"}
	if *flagCompilerFlags != "" {
		args = append(args, "-gcflags", *flagCompilerFlags)
	}
	args = append(args, r.pkgs...)
	cmd := exec.Command(*flagGoCmd, args...)
	cmd.Dir = filepath.Join(goroot, "src")
	return runCmd(name, cmd)
}

type size struct {
	// path is either a path to a file ("$GOROOT/test/helloworld.go") or a package path ("cmd/go").
	path   string
	isLong bool
}

func (r size) long() bool { return r.isLong }

func (r size) run(name string, count int) error {
	if strings.HasPrefix(r.path, "$GOROOT/") {
		r.path = goroot + "/" + r.path[len("$GOROOT/"):]
	}

	cmd := exec.Command(*flagGoCmd, "build", "-o", "_compilebenchout_", r.path)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return err
	}
	defer os.Remove("_compilebenchout_")
	info, err := os.Stat("_compilebenchout_")
	if err != nil {
		return err
	}
	out, err := exec.Command("size", "_compilebenchout_").CombinedOutput()
	if err != nil {
		return fmt.Errorf("size: %v\n%s", err, out)
	}
	lines := strings.Split(string(out), "\n")
	if len(lines) < 2 {
		return fmt.Errorf("not enough output from size: %s", out)
	}
	f := strings.Fields(lines[1])
	if strings.HasPrefix(lines[0], "__TEXT") && len(f) >= 2 { // OS X
		fmt.Printf("%s 1 %s text-bytes %s data-bytes %v exe-bytes\n", name, f[0], f[1], info.Size())
	} else if strings.Contains(lines[0], "bss") && len(f) >= 3 {
		fmt.Printf("%s 1 %s text-bytes %s data-bytes %s bss-bytes %v exe-bytes\n", name, f[0], f[1], f[2], info.Size())
	}
	return nil
}

type compile struct{ dir string }

func (compile) long() bool { return false }

func (c compile) run(name string, count int) error {
	// Make sure dependencies needed by go tool compile are built.
	out, err := exec.Command(*flagGoCmd, "build", c.dir).CombinedOutput()
	if err != nil {
		return fmt.Errorf("go build %s: %v\n%s", c.dir, err, out)
	}

	// Find dir and source file list.
	pkg, err := goList(c.dir)
	if err != nil {
		return err
	}

	importcfg, err := genImportcfgFile(c.dir, false)
	if err != nil {
		return err
	}

	// If this package has assembly files, we'll need to pass a symabis
	// file to the compiler; call a helper to invoke the assembler
	// to do that.
	var symAbisFile string
	var asmIncFile string
	if len(pkg.SFiles) != 0 {
		symAbisFile = filepath.Join(pkg.Dir, "symabis")
		asmIncFile = filepath.Join(pkg.Dir, "go_asm.h")
		content := "\n"
		if err := os.WriteFile(asmIncFile, []byte(content), 0666); err != nil {
			return fmt.Errorf("os.WriteFile(%s) failed: %v", asmIncFile, err)
		}
		defer os.Remove(symAbisFile)
		defer os.Remove(asmIncFile)
		if err := genSymAbisFile(pkg, symAbisFile, pkg.Dir); err != nil {
			return err
		}
	}

	args := []string{"-o", "_compilebench_.o", "-p", pkg.ImportPath}
	args = append(args, strings.Fields(*flagCompilerFlags)...)
	if symAbisFile != "" {
		args = append(args, "-symabis", symAbisFile)
	}
	if importcfg != "" {
		args = append(args, "-importcfg", importcfg)
		defer os.Remove(importcfg)
	}
	args = append(args, pkg.GoFiles...)
	if err := runBuildCmd(name, count, pkg.Dir, compiler, args); err != nil {
		return err
	}

	opath := pkg.Dir + "/_compilebench_.o"
	if *flagObj {
		// TODO(josharian): object files are big; just read enough to find what we seek.
		data, err := os.ReadFile(opath)
		if err != nil {
			log.Print(err)
		}
		// Find start of export data.
		i := bytes.Index(data, []byte("\n$$B\n")) + len("\n$$B\n")
		// Count bytes to end of export data.
		nexport := bytes.Index(data[i:], []byte("\n$$\n"))
		fmt.Printf(" %d object-bytes %d export-bytes", len(data), nexport)
	}
	fmt.Println()

	os.Remove(opath)
	return nil
}

type link struct{ dir, flags string }

func (link) long() bool { return false }

func (r link) run(name string, count int) error {
	if linker == "" {
		// No linker. Skip the test.
		return nil
	}

	// Build dependencies.
	out, err := exec.Command(*flagGoCmd, "build", "-o", "/dev/null", r.dir).CombinedOutput()
	if err != nil {
		return fmt.Errorf("go build -a %s: %v\n%s", r.dir, err, out)
	}

	importcfg, err := genImportcfgFile(r.dir, true)
	if err != nil {
		return err
	}
	defer os.Remove(importcfg)

	// Build the main package.
	pkg, err := goList(r.dir)
	if err != nil {
		return err
	}
	args := []string{"-o", "_compilebench_.o", "-importcfg", importcfg}
	args = append(args, pkg.GoFiles...)
	if *flagTrace {
		fmt.Fprintf(os.Stderr, "running: %s %+v\n",
			compiler, args)
	}
	cmd := exec.Command(compiler, args...)
	cmd.Dir = pkg.Dir
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	err = cmd.Run()
	if err != nil {
		return fmt.Errorf("compiling: %v", err)
	}
	defer os.Remove(pkg.Dir + "/_compilebench_.o")

	// Link the main package.
	args = []string{"-o", "_compilebench_.exe", "-importcfg", importcfg}
	args = append(args, strings.Fields(*flagLinkerFlags)...)
	args = append(args, strings.Fields(r.flags)...)
	args = append(args, "_compilebench_.o")
	if err := runBuildCmd(name, count, pkg.Dir, linker, args); err != nil {
		return err
	}
	fmt.Println()
	defer os.Remove(pkg.Dir + "/_compilebench_.exe")

	return err
}

// runBuildCmd runs "tool args..." in dir, measures standard build
// tool metrics, and prints a benchmark line. The caller may print
// additional metrics and then must print a newline.
//
// This assumes tool accepts standard build tool flags like
// -memprofilerate, -memprofile, and -cpuprofile.
func runBuildCmd(name string, count int, dir, tool string, args []string) error {
	var preArgs []string
	if *flagMemprofilerate >= 0 {
		preArgs = append(preArgs, "-memprofilerate", fmt.Sprint(*flagMemprofilerate))
	}
	if *flagAlloc || *flagCpuprofile != "" || *flagMemprofile != "" {
		if *flagAlloc || *flagMemprofile != "" {
			preArgs = append(preArgs, "-memprofile", "_compilebench_.memprof")
		}
		if *flagCpuprofile != "" {
			preArgs = append(preArgs, "-cpuprofile", "_compilebench_.cpuprof")
		}
	}
	if *flagTrace {
		fmt.Fprintf(os.Stderr, "running: %s %+v\n",
			tool, append(preArgs, args...))
	}
	cmd := exec.Command(tool, append(preArgs, args...)...)
	cmd.Dir = dir
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	start := time.Now()
	err := cmd.Run()
	if err != nil {
		return err
	}
	end := time.Now()

	haveAllocs, haveRSS := false, false
	var allocs, allocbytes, rssbytes int64
	if *flagAlloc || *flagMemprofile != "" {
		out, err := os.ReadFile(dir + "/_compilebench_.memprof")
		if err != nil {
			log.Print("cannot find memory profile after compilation")
		}
		for _, line := range strings.Split(string(out), "\n") {
			f := strings.Fields(line)
			if len(f) < 4 || f[0] != "#" || f[2] != "=" {
				continue
			}
			val, err := strconv.ParseInt(f[3], 0, 64)
			if err != nil {
				continue
			}
			haveAllocs = true
			switch f[1] {
			case "TotalAlloc":
				allocbytes = val
			case "Mallocs":
				allocs = val
			case "MaxRSS":
				haveRSS = true
				rssbytes = val
			}
		}
		if !haveAllocs {
			log.Println("missing stats in memprof (golang.org/issue/18641)")
		}

		if *flagMemprofile != "" {
			outpath := *flagMemprofile
			if *flagCount != 1 {
				outpath = fmt.Sprintf("%s_%d", outpath, count)
			}
			if err := os.WriteFile(outpath, out, 0666); err != nil {
				log.Print(err)
			}
		}
		os.Remove(dir + "/_compilebench_.memprof")
	}

	if *flagCpuprofile != "" {
		out, err := os.ReadFile(dir + "/_compilebench_.cpuprof")
		if err != nil {
			log.Print(err)
		}
		outpath := *flagCpuprofile
		if *flagCount != 1 {
			outpath = fmt.Sprintf("%s_%d", outpath, count)
		}
		if err := os.WriteFile(outpath, out, 0666); err != nil {
			log.Print(err)
		}
		os.Remove(dir + "/_compilebench_.cpuprof")
	}

	wallns := end.Sub(start).Nanoseconds()
	userns := cmd.ProcessState.UserTime().Nanoseconds()

	fmt.Printf("%s 1 %d ns/op %d user-ns/op", name, wallns, userns)
	if haveAllocs {
		fmt.Printf(" %d B/op %d allocs/op", allocbytes, allocs)
	}
	if haveRSS {
		fmt.Printf(" %d maxRSS/op", rssbytes)
	}

	return nil
}

func checkCompilingRuntimeFlag(assembler string) error {
	td, err := os.MkdirTemp("", "asmsrcd")
	if err != nil {
		return fmt.Errorf("MkdirTemp failed: %v", err)
	}
	defer os.RemoveAll(td)
	src := filepath.Join(td, "asm.s")
	obj := filepath.Join(td, "asm.o")
	const code = `
TEXT ·foo(SB),$0-0
RET
`
	if err := os.WriteFile(src, []byte(code), 0644); err != nil {
		return fmt.Errorf("writing %s failed: %v", src, err)
	}

	// Try compiling the assembly source file passing
	// -compiling-runtime; if it succeeds, then we'll need it
	// when doing assembly of the reflect package later on.
	// If it does not succeed, the assumption is that it's not
	// needed.
	args := []string{"-o", obj, "-p", "reflect", "-compiling-runtime", src}
	cmd := exec.Command(assembler, args...)
	cmd.Dir = td
	out, aerr := cmd.CombinedOutput()
	if aerr != nil {
		if strings.Contains(string(out), "flag provided but not defined: -compiling-runtime") {
			// flag not defined: assume we're using a recent assembler, so
			// don't use -compiling-runtime.
			return nil
		}
		// error is not flag-related; report it.
		return fmt.Errorf("problems invoking assembler with args %+v: error %v\n%s\n", args, aerr, out)
	}
	// asm invocation succeeded -- assume we need the flag.
	needCompilingRuntimeFlag = true
	return nil
}

// genSymAbisFile runs the assembler on the target package asm files
// with "-gensymabis" to produce a symabis file that will feed into
// the Go source compilation. This is fairly hacky in that if the
// asm invocation convention changes it will need to be updated
// (hopefully that will not be needed too frequently).
func genSymAbisFile(pkg *Pkg, symAbisFile, incdir string) error {
	args := []string{"-gensymabis", "-o", symAbisFile,
		"-p", pkg.ImportPath,
		"-I", filepath.Join(goroot, "pkg", "include"),
		"-I", incdir,
		"-D", "GOOS_" + runtime.GOOS,
		"-D", "GOARCH_" + runtime.GOARCH}
	if pkg.ImportPath == "reflect" && needCompilingRuntimeFlag {
		args = append(args, "-compiling-runtime")
	}
	args = append(args, pkg.SFiles...)
	if *flagTrace {
		fmt.Fprintf(os.Stderr, "running: %s %+v\n",
			assembler, args)
	}
	cmd := exec.Command(assembler, args...)
	cmd.Dir = pkg.Dir
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		return fmt.Errorf("assembling to produce symabis file: %v", err)
	}
	return nil
}

// genImportcfgFile generates an importcfg file for building package
// dir. Returns the generated importcfg file path (or empty string
// if the package has no dependency).
func genImportcfgFile(dir string, full bool) (string, error) {
	need := "{{.Imports}}"
	if full {
		// for linking, we need transitive dependencies
		need = "{{.Deps}}"
	}

	// find imported/dependent packages
	cmd := exec.Command(*flagGoCmd, "list", "-f", need, dir)
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("go list -f %s %s: %v", need, dir, err)
	}
	// trim [ ]\n
	if len(out) < 3 || out[0] != '[' || out[len(out)-2] != ']' || out[len(out)-1] != '\n' {
		return "", fmt.Errorf("unexpected output from go list -f %s %s: %s", need, dir, out)
	}
	out = out[1 : len(out)-2]
	if len(out) == 0 {
		return "", nil
	}

	// build importcfg for imported packages
	cmd = exec.Command(*flagGoCmd, "list", "-export", "-f", "{{if .Export}}packagefile {{.ImportPath}}={{.Export}}{{end}}")
	cmd.Args = append(cmd.Args, strings.Fields(string(out))...)
	cmd.Stderr = os.Stderr
	out, err = cmd.Output()
	if err != nil {
		return "", fmt.Errorf("generating importcfg for %s: %s: %v", dir, cmd, err)
	}

	f, err := os.CreateTemp("", "importcfg")
	if err != nil {
		return "", fmt.Errorf("creating tmp importcfg file failed: %v", err)
	}
	defer f.Close()
	if _, err := f.Write(out); err != nil {
		return "", fmt.Errorf("writing importcfg file %s failed: %v", f.Name(), err)
	}
	return f.Name(), nil
}
