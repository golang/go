// Copyright 2015 The Go Authors.  All rights reserved.
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
// allocated (see ``go doc runtime.MemProfileRate'').
// Lowering the rate (for example, -memprofilerate 64000) produces
// a more fine-grained and therefore accurate profile, but it also incurs
// execution cost. For benchmark comparisons, never use timings
// obtained with a low -memprofilerate option.
//
// Example
//
// Assuming the base version of the compiler has been saved with
// ``toolstash save,'' this sequence compares the old and new compiler:
//
//	compilebench -count 10 -compile $(toolstash -n compile) >old.txt
//	compilebench -count 10 >new.txt
//	benchstat old.txt new.txt
//
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

var (
	goroot   string
	compiler string
	linker   string
	runRE    *regexp.Regexp
	is6g     bool
)

var (
	flagGoCmd          = flag.String("go", "go", "path to \"go\" command")
	flagAlloc          = flag.Bool("alloc", false, "report allocations")
	flagObj            = flag.Bool("obj", false, "report object file stats")
	flagCompiler       = flag.String("compile", "", "use `exe` as the cmd/compile binary")
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
	Dir     string
	GoFiles []string
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
	// Make sure dependencies needed by go tool compile are installed to GOROOT/pkg.
	out, err := exec.Command(*flagGoCmd, "build", "-i", c.dir).CombinedOutput()
	if err != nil {
		return fmt.Errorf("go build -i %s: %v\n%s", c.dir, err, out)
	}

	// Find dir and source file list.
	pkg, err := goList(c.dir)
	if err != nil {
		return err
	}

	args := []string{"-o", "_compilebench_.o"}
	args = append(args, strings.Fields(*flagCompilerFlags)...)
	args = append(args, pkg.GoFiles...)
	if err := runBuildCmd(name, count, pkg.Dir, compiler, args); err != nil {
		return err
	}

	opath := pkg.Dir + "/_compilebench_.o"
	if *flagObj {
		// TODO(josharian): object files are big; just read enough to find what we seek.
		data, err := ioutil.ReadFile(opath)
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
	out, err := exec.Command(*flagGoCmd, "build", "-i", "-o", "/dev/null", r.dir).CombinedOutput()
	if err != nil {
		return fmt.Errorf("go build -i %s: %v\n%s", r.dir, err, out)
	}

	// Build the main package.
	pkg, err := goList(r.dir)
	if err != nil {
		return err
	}
	args := []string{"-o", "_compilebench_.o"}
	args = append(args, pkg.GoFiles...)
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
	args = []string{"-o", "_compilebench_.exe"}
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
		out, err := ioutil.ReadFile(dir + "/_compilebench_.memprof")
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
			if err := ioutil.WriteFile(outpath, out, 0666); err != nil {
				log.Print(err)
			}
		}
		os.Remove(dir + "/_compilebench_.memprof")
	}

	if *flagCpuprofile != "" {
		out, err := ioutil.ReadFile(dir + "/_compilebench_.cpuprof")
		if err != nil {
			log.Print(err)
		}
		outpath := *flagCpuprofile
		if *flagCount != 1 {
			outpath = fmt.Sprintf("%s_%d", outpath, count)
		}
		if err := ioutil.WriteFile(outpath, out, 0666); err != nil {
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
