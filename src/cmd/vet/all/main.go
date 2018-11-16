// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// The vet/all command runs go vet on the standard library and commands.
// It compares the output against a set of whitelists
// maintained in the whitelist directory.
//
// This program attempts to build packages from golang.org/x/tools,
// which must be in your GOPATH.
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"go/build"
	"go/types"
	"internal/testenv"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
)

var (
	flagPlatforms = flag.String("p", "", "platform(s) to use e.g. linux/amd64,darwin/386")
	flagAll       = flag.Bool("all", false, "run all platforms")
	flagNoLines   = flag.Bool("n", false, "don't print line numbers")
)

var cmdGoPath string
var failed uint32 // updated atomically

func main() {
	log.SetPrefix("vet/all: ")
	log.SetFlags(0)

	var err error
	cmdGoPath, err = testenv.GoTool()
	if err != nil {
		log.Print("could not find cmd/go; skipping")
		// We're on a platform that can't run cmd/go.
		// We want this script to be able to run as part of all.bash,
		// so return cleanly rather than with exit code 1.
		return
	}

	flag.Parse()
	switch {
	case *flagAll && *flagPlatforms != "":
		log.Print("-all and -p flags are incompatible")
		flag.Usage()
		os.Exit(2)
	case *flagPlatforms != "":
		vetPlatforms(parseFlagPlatforms())
	case *flagAll:
		vetPlatforms(allPlatforms())
	default:
		hostPlatform.vet()
	}
	if atomic.LoadUint32(&failed) != 0 {
		os.Exit(1)
	}
}

var hostPlatform = platform{os: build.Default.GOOS, arch: build.Default.GOARCH}

func allPlatforms() []platform {
	var pp []platform
	cmd := exec.Command(cmdGoPath, "tool", "dist", "list")
	out, err := cmd.Output()
	if err != nil {
		log.Fatal(err)
	}
	lines := bytes.Split(out, []byte{'\n'})
	for _, line := range lines {
		if len(line) == 0 {
			continue
		}
		pp = append(pp, parsePlatform(string(line)))
	}
	return pp
}

func parseFlagPlatforms() []platform {
	var pp []platform
	components := strings.Split(*flagPlatforms, ",")
	for _, c := range components {
		pp = append(pp, parsePlatform(c))
	}
	return pp
}

func parsePlatform(s string) platform {
	vv := strings.Split(s, "/")
	if len(vv) != 2 {
		log.Fatalf("could not parse platform %s, must be of form goos/goarch", s)
	}
	return platform{os: vv[0], arch: vv[1]}
}

type whitelist map[string]int

// load adds entries from the whitelist file, if present, for os/arch to w.
func (w whitelist) load(goos string, goarch string) {
	sz := types.SizesFor("gc", goarch)
	if sz == nil {
		log.Fatalf("unknown type sizes for arch %q", goarch)
	}
	archbits := 8 * sz.Sizeof(types.Typ[types.UnsafePointer])

	// Look up whether goarch has a shared arch suffix,
	// such as mips64x for mips64 and mips64le.
	archsuff := goarch
	if x, ok := archAsmX[goarch]; ok {
		archsuff = x
	}

	// Load whitelists.
	filenames := []string{
		"all.txt",
		goos + ".txt",
		goarch + ".txt",
		goos + "_" + goarch + ".txt",
		fmt.Sprintf("%dbit.txt", archbits),
	}
	if goarch != archsuff {
		filenames = append(filenames,
			archsuff+".txt",
			goos+"_"+archsuff+".txt",
		)
	}

	// We allow error message templates using GOOS and GOARCH.
	if goos == "android" {
		goos = "linux" // so many special cases :(
	}

	// Read whitelists and do template substitution.
	replace := strings.NewReplacer("GOOS", goos, "GOARCH", goarch, "ARCHSUFF", archsuff)

	for _, filename := range filenames {
		path := filepath.Join("whitelist", filename)
		f, err := os.Open(path)
		if err != nil {
			// Allow not-exist errors; not all combinations have whitelists.
			if os.IsNotExist(err) {
				continue
			}
			log.Fatal(err)
		}
		scan := bufio.NewScanner(f)
		for scan.Scan() {
			line := scan.Text()
			if len(line) == 0 || strings.HasPrefix(line, "//") {
				continue
			}
			w[replace.Replace(line)]++
		}
		if err := scan.Err(); err != nil {
			log.Fatal(err)
		}
	}
}

type platform struct {
	os   string
	arch string
}

func (p platform) String() string {
	return p.os + "/" + p.arch
}

// ignorePathPrefixes are file path prefixes that should be ignored wholesale.
var ignorePathPrefixes = [...]string{
	// These testdata dirs have lots of intentionally broken/bad code for tests.
	"cmd/go/testdata/",
	"cmd/vet/testdata/",
	"go/printer/testdata/",
}

func vetPlatforms(pp []platform) {
	for _, p := range pp {
		p.vet()
	}
}

func (p platform) vet() {
	if p.os == "linux" && (p.arch == "riscv64" || p.arch == "sparc64") {
		// TODO(tklauser): enable as soon as these ports have fully landed
		fmt.Printf("skipping %s/%s\n", p.os, p.arch)
		return
	}

	if p.os == "windows" && p.arch == "arm" {
		// TODO(jordanrh1): enable as soon as the windows/arm port has fully landed
		fmt.Println("skipping windows/arm")
		return
	}

	if p.os == "aix" && p.arch == "ppc64" {
		// TODO(aix): enable as soon as the aix/ppc64 port has fully landed
		fmt.Println("skipping aix/ppc64")
		return
	}

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "go run main.go -p %s\n", p)

	// Load whitelist(s).
	w := make(whitelist)
	w.load(p.os, p.arch)

	tmpdir, err := ioutil.TempDir("", "cmd-vet-all")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	// Build the go/packages-based vet command from the x/tools
	// repo. It is considerably faster than "go vet", which rebuilds
	// the standard library.
	vetTool := filepath.Join(tmpdir, "vet")
	cmd := exec.Command(cmdGoPath, "build", "-o", vetTool, "golang.org/x/tools/go/analysis/cmd/vet")
	cmd.Dir = filepath.Join(runtime.GOROOT(), "src")
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stderr
	if err := cmd.Run(); err != nil {
		log.Fatal(err)
	}

	// TODO: The unsafeptr checks are disabled for now,
	// because there are so many false positives,
	// and no clear way to improve vet to eliminate large chunks of them.
	// And having them in the whitelists will just cause annoyance
	// and churn when working on the runtime.
	cmd = exec.Command(vetTool,
		"-unsafeptr=0",
		"-nilness=0", // expensive, uses SSA
		"std",
		"cmd/...",
		"cmd/compile/internal/gc/testdata",
	)
	cmd.Dir = filepath.Join(runtime.GOROOT(), "src")
	cmd.Env = append(os.Environ(), "GOOS="+p.os, "GOARCH="+p.arch, "CGO_ENABLED=0")
	stderr, err := cmd.StderrPipe()
	if err != nil {
		log.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		log.Fatal(err)
	}

	// Process vet output.
	scan := bufio.NewScanner(stderr)
	var parseFailed bool
NextLine:
	for scan.Scan() {
		line := scan.Text()
		if strings.HasPrefix(line, "vet: ") {
			// Typecheck failure: Malformed syntax or multiple packages or the like.
			// This will yield nicer error messages elsewhere, so ignore them here.

			// This includes warnings from asmdecl of the form:
			//   "vet: foo.s:16: [amd64] cannot check cross-package assembly function"
			continue
		}

		if strings.HasPrefix(line, "panic: ") {
			// Panic in vet. Don't filter anything, we want the complete output.
			parseFailed = true
			fmt.Fprintf(os.Stderr, "panic in vet (to reproduce: go run main.go -p %s):\n", p)
			fmt.Fprintln(os.Stderr, line)
			io.Copy(os.Stderr, stderr)
			break
		}
		if strings.HasPrefix(line, "# ") {
			// 'go vet' prefixes the output of each vet invocation by a comment:
			//    # [package]
			continue
		}

		// Parse line.
		// Assume the part before the first ": "
		// is the "file:line:col: " information.
		// TODO(adonovan): parse vet -json output.
		var file, lineno, msg string
		if i := strings.Index(line, ": "); i >= 0 {
			msg = line[i+len(": "):]

			words := strings.Split(line[:i], ":")
			switch len(words) {
			case 3:
				_ = words[2] // ignore column
				fallthrough
			case 2:
				lineno = words[1]
				fallthrough
			case 1:
				file = words[0]

				// Make the file name relative to GOROOT/src.
				if rel, err := filepath.Rel(cmd.Dir, file); err == nil {
					file = rel
				}
			default:
				// error: too many columns
			}
		}
		if file == "" {
			if !parseFailed {
				parseFailed = true
				fmt.Fprintf(os.Stderr, "failed to parse %s vet output:\n", p)
			}
			fmt.Fprintln(os.Stderr, line)
			continue
		}

		msg = strings.TrimSpace(msg)

		for _, ignore := range ignorePathPrefixes {
			if strings.HasPrefix(file, filepath.FromSlash(ignore)) {
				continue NextLine
			}
		}

		key := file + ": " + msg
		if w[key] == 0 {
			// Vet error with no match in the whitelist. Print it.
			if *flagNoLines {
				fmt.Fprintf(&buf, "%s: %s\n", file, msg)
			} else {
				fmt.Fprintf(&buf, "%s:%s: %s\n", file, lineno, msg)
			}
			atomic.StoreUint32(&failed, 1)
			continue
		}
		w[key]--
	}
	if parseFailed {
		atomic.StoreUint32(&failed, 1)
		return
	}
	if scan.Err() != nil {
		log.Fatalf("failed to scan vet output: %v", scan.Err())
	}
	err = cmd.Wait()
	// We expect vet to fail.
	// Make sure it has failed appropriately, though (for example, not a PathError).
	if _, ok := err.(*exec.ExitError); !ok {
		log.Fatalf("unexpected go vet execution failure: %v", err)
	}
	printedHeader := false
	if len(w) > 0 {
		for k, v := range w {
			if v != 0 {
				if !printedHeader {
					fmt.Fprintln(&buf, "unmatched whitelist entries:")
					printedHeader = true
				}
				for i := 0; i < v; i++ {
					fmt.Fprintln(&buf, k)
				}
				atomic.StoreUint32(&failed, 1)
			}
		}
	}

	os.Stdout.Write(buf.Bytes())
}

// archAsmX maps architectures to the suffix usually used for their assembly files,
// if different than the arch name itself.
var archAsmX = map[string]string{
	"android":  "linux",
	"mips64":   "mips64x",
	"mips64le": "mips64x",
	"mips":     "mipsx",
	"mipsle":   "mipsx",
	"ppc64":    "ppc64x",
	"ppc64le":  "ppc64x",
}
