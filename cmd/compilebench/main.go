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
//	-count n
//		Run each benchmark n times (default 1).
//
//	-cpuprofile file
//		Write a CPU profile of the compiler to file.
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
//  -pkg
//		Benchmark compiling a single package.
//
//	-run regexp
//		Only run benchmarks with names matching regexp.
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
	"flag"
	"fmt"
	"go/build"
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
	runRE    *regexp.Regexp
	is6g     bool
)

var (
	flagGoCmd          = flag.String("go", "go", "path to \"go\" command")
	flagAlloc          = flag.Bool("alloc", false, "report allocations")
	flagObj            = flag.Bool("obj", false, "report object file stats")
	flagCompiler       = flag.String("compile", "", "use `exe` as the cmd/compile binary")
	flagCompilerFlags  = flag.String("compileflags", "", "additional `flags` to pass to compile")
	flagRun            = flag.String("run", "", "run benchmarks matching `regexp`")
	flagCount          = flag.Int("count", 1, "run benchmarks `n` times")
	flagCpuprofile     = flag.String("cpuprofile", "", "write CPU profile to `file`")
	flagMemprofile     = flag.String("memprofile", "", "write memory profile to `file`")
	flagMemprofilerate = flag.Int64("memprofilerate", -1, "set memory profile `rate`")
	flagPackage        = flag.String("pkg", "", "if set, benchmark the package at path `pkg`")
	flagShort          = flag.Bool("short", false, "skip long-running benchmarks")
)

var tests = []struct {
	name string
	dir  string
	long bool
}{
	{"BenchmarkTemplate", "html/template", false},
	{"BenchmarkUnicode", "unicode", false},
	{"BenchmarkGoTypes", "go/types", false},
	{"BenchmarkCompiler", "cmd/compile/internal/gc", false},
	{"BenchmarkSSA", "cmd/compile/internal/ssa", false},
	{"BenchmarkFlate", "compress/flate", false},
	{"BenchmarkGoParser", "go/parser", false},
	{"BenchmarkReflect", "reflect", false},
	{"BenchmarkTar", "archive/tar", false},
	{"BenchmarkXML", "encoding/xml", false},
	{"BenchmarkStdCmd", "", true},
	{"BenchmarkHelloSize", "", false},
	{"BenchmarkCmdGoSize", "", true},
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

	compiler = *flagCompiler
	if compiler == "" {
		out, err := exec.Command(*flagGoCmd, "tool", "-n", "compile").CombinedOutput()
		if err != nil {
			out, err = exec.Command(*flagGoCmd, "tool", "-n", "6g").CombinedOutput()
			is6g = true
			if err != nil {
				out, err = exec.Command(*flagGoCmd, "tool", "-n", "compile").CombinedOutput()
				log.Fatalf("go tool -n compiler: %v\n%s", err, out)
			}
		}
		compiler = strings.TrimSpace(string(out))
	}

	if *flagRun != "" {
		r, err := regexp.Compile(*flagRun)
		if err != nil {
			log.Fatalf("invalid -run argument: %v", err)
		}
		runRE = r
	}

	for i := 0; i < *flagCount; i++ {
		if *flagPackage != "" {
			runBuild("BenchmarkPkg", *flagPackage, i)
			continue
		}
		for _, tt := range tests {
			if tt.long && *flagShort {
				continue
			}
			if runRE == nil || runRE.MatchString(tt.name) {
				runBuild(tt.name, tt.dir, i)
			}
		}
	}
}

func runCmd(name string, cmd *exec.Cmd) {
	start := time.Now()
	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("%v: %v\n%s", name, err, out)
		return
	}
	fmt.Printf("%s 1 %d ns/op\n", name, time.Since(start).Nanoseconds())
}

func runStdCmd() {
	args := []string{"build", "-a"}
	if *flagCompilerFlags != "" {
		args = append(args, "-gcflags", *flagCompilerFlags)
	}
	args = append(args, "std", "cmd")
	cmd := exec.Command(*flagGoCmd, args...)
	cmd.Dir = filepath.Join(goroot, "src")
	runCmd("BenchmarkStdCmd", cmd)
}

// path is either a path to a file ("$GOROOT/test/helloworld.go") or a package path ("cmd/go").
func runSize(name, path string) {
	cmd := exec.Command(*flagGoCmd, "build", "-o", "_compilebenchout_", path)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		log.Print(err)
		return
	}
	defer os.Remove("_compilebenchout_")
	info, err := os.Stat("_compilebenchout_")
	if err != nil {
		log.Print(err)
		return
	}
	out, err := exec.Command("size", "_compilebenchout_").CombinedOutput()
	if err != nil {
		log.Printf("size: %v\n%s", err, out)
		return
	}
	lines := strings.Split(string(out), "\n")
	if len(lines) < 2 {
		log.Printf("not enough output from size: %s", out)
		return
	}
	f := strings.Fields(lines[1])
	if strings.HasPrefix(lines[0], "__TEXT") && len(f) >= 2 { // OS X
		fmt.Printf("%s 1 %s text-bytes %s data-bytes %v exe-bytes\n", name, f[0], f[1], info.Size())
	} else if strings.Contains(lines[0], "bss") && len(f) >= 3 {
		fmt.Printf("%s 1 %s text-bytes %s data-bytes %s bss-bytes %v exe-bytes\n", name, f[0], f[1], f[2], info.Size())
	}
}

func runBuild(name, dir string, count int) {
	switch name {
	case "BenchmarkStdCmd":
		runStdCmd()
		return
	case "BenchmarkCmdGoSize":
		runSize("BenchmarkCmdGoSize", "cmd/go")
		return
	case "BenchmarkHelloSize":
		runSize("BenchmarkHelloSize", filepath.Join(goroot, "test/helloworld.go"))
		return
	}

	pkg, err := build.Import(dir, ".", 0)
	if err != nil {
		log.Print(err)
		return
	}
	args := []string{"-o", "_compilebench_.o"}
	if is6g {
		*flagMemprofilerate = -1
		*flagAlloc = false
		*flagCpuprofile = ""
		*flagMemprofile = ""
	}
	if *flagMemprofilerate >= 0 {
		args = append(args, "-memprofilerate", fmt.Sprint(*flagMemprofilerate))
	}
	args = append(args, strings.Fields(*flagCompilerFlags)...)
	if *flagAlloc || *flagCpuprofile != "" || *flagMemprofile != "" {
		if *flagAlloc || *flagMemprofile != "" {
			args = append(args, "-memprofile", "_compilebench_.memprof")
		}
		if *flagCpuprofile != "" {
			args = append(args, "-cpuprofile", "_compilebench_.cpuprof")
		}
	}
	args = append(args, pkg.GoFiles...)
	cmd := exec.Command(compiler, args...)
	cmd.Dir = pkg.Dir
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	start := time.Now()
	err = cmd.Run()
	if err != nil {
		log.Printf("%v: %v", name, err)
		return
	}
	end := time.Now()

	var allocs, allocbytes int64
	if *flagAlloc || *flagMemprofile != "" {
		out, err := ioutil.ReadFile(pkg.Dir + "/_compilebench_.memprof")
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
			switch f[1] {
			case "TotalAlloc":
				allocbytes = val
			case "Mallocs":
				allocs = val
			}
		}

		if *flagMemprofile != "" {
			if err := ioutil.WriteFile(*flagMemprofile, out, 0666); err != nil {
				log.Print(err)
			}
		}
		os.Remove(pkg.Dir + "/_compilebench_.memprof")
	}

	if *flagCpuprofile != "" {
		out, err := ioutil.ReadFile(pkg.Dir + "/_compilebench_.cpuprof")
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
		os.Remove(pkg.Dir + "/_compilebench_.cpuprof")
	}

	wallns := end.Sub(start).Nanoseconds()
	userns := cmd.ProcessState.UserTime().Nanoseconds()

	fmt.Printf("%s 1 %d ns/op %d user-ns/op", name, wallns, userns)
	if *flagAlloc {
		fmt.Printf(" %d B/op %d allocs/op", allocbytes, allocs)
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
}
