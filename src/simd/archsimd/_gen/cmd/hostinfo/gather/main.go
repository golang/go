// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"log"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"simd/archsimd/_gen/gentools"
)

var printFlag = flag.Bool("print", false, "Print reproduction steps without executing them")

var (
	hostinfoDir string
	goroot      string
	workDir     string
)

// shellQuote quotes a single token for a UNIX shell.
func shellQuote(s string) string {
	if s == "" {
		return "''"
	}
	safe := true
	for _, r := range s {
		if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') ||
			r == '_' || r == '-' || r == '.' || r == '/' || r == '=' || r == ':' || r == ',') {
			safe = false
			break
		}
	}
	if safe {
		return s
	}
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

func run(args ...string) {
	if len(args) == 0 {
		return
	}
	if *printFlag {
		words := make([]string, len(args))
		for i, a := range args {
			words[i] = shellQuote(a)
		}
		fmt.Println(strings.Join(words, " "))
		return
	}

	var env []string
	for len(args) > 0 && strings.Contains(args[0], "=") {
		env = append(env, args[0])
		args = args[1:]
	}
	if len(args) == 0 {
		return
	}

	c := exec.Command(args[0], args[1:]...)
	if len(env) > 0 {
		c.Env = append(os.Environ(), env...)
	}
	c.Stdin, c.Stdout, c.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := c.Run(); err != nil {
		log.Fatalf("%s failed: %v", args[0], err)
	}
}

func compile(goos, goarch string) string {
	bin := filepath.Join(workDir, fmt.Sprintf("hostinfo-%s-%s", goos, goarch))
	run("GOOS="+goos, "GOARCH="+goarch, "GOEXPERIMENT=simd", "go", "build", "-C", hostinfoDir, "-o", bin, ".")
	return bin
}

func runGomote(builder, goarch string) {
	bin := compile("linux", goarch)
	run("GOMOTE_GROUP=simd", "gomote", "create", builder)
	run("gomote", "-group", "simd", "put", bin, "hostinfo")
	run("gomote", "-group", "simd", "run", "./hostinfo", "-host", builder)
	run("gomote", "-group", "simd", "destroy")
}

func runQEMUarm64(name, cpu string) {
	bin := compile("linux", "arm64")
	run("qemu-aarch64", "-cpu", cpu, bin, "-host", name)
}

var hosts = map[string]func(){
	"local": func() {
		run("GOEXPERIMENT=simd", "go", "run", "-C", hostinfoDir, ".", "-host", "local")
	},
	"gotip-linux-amd64_avx512": func() {
		runGomote("gotip-linux-amd64_avx512", "amd64")
	},
	"gotip-linux-amd64": func() {
		runGomote("gotip-linux-amd64", "amd64")
	},
	"gotip-linux-arm64": func() {
		runGomote("gotip-linux-arm64", "arm64")
	},
	"qemu-aarch64-neon": func() {
		runQEMUarm64("qemu-aarch64-neon", "neoverse-n1")
	},
	"qemu-aarch64-sve128": func() {
		runQEMUarm64("qemu-aarch64-sve128", "neoverse-n2")
	},
	"qemu-aarch64-sve256": func() {
		runQEMUarm64("qemu-aarch64-sve256", "neoverse-v1")
	},
	"wasip1-wasmtime": func() {
		bin := compile("wasip1", "wasm")
		run("wasmtime", "run", bin, "-host", "wasip1-wasmtime")
	},
	"wasip1-wazero": func() {
		bin := compile("wasip1", "wasm")
		run("wazero", "run", bin, "-host", "wasip1-wazero")
	},
	"js-node": func() {
		bin := compile("js", "wasm")
		run("node", filepath.Join(goroot, "lib/wasm/wasm_exec_node.js"), bin, "-host", "js-node")
	},
}

func usage() {
	fmt.Printf("Usage: gather [-print] <all | host ...>\n\n")
	fmt.Println("Known pseudo-hosts:")
	fmt.Println("  - all (all known pseudo-hosts; must appear alone)")
	for _, host := range slices.Sorted(maps.Keys(hosts)) {
		fmt.Printf("  - %s\n", host)
	}
	fmt.Println("\nFlags:")
	flag.PrintDefaults()
}

func main() {
	flag.Usage = func() {
		usage()
		os.Exit(0)
	}
	flag.Parse()

	args := flag.Args()
	if len(args) == 0 {
		usage()
		os.Exit(1)
	}

	var targets []string
	if args[0] == "all" {
		if len(args) > 1 {
			fmt.Fprintln(os.Stderr, "Error: 'all' must appear alone")
			os.Exit(1)
		}
		targets = slices.Sorted(maps.Keys(hosts))
	} else {
		for _, arg := range args {
			if arg == "all" {
				fmt.Fprintln(os.Stderr, "Error: 'all' must appear alone")
				os.Exit(1)
			}
			if _, ok := hosts[arg]; !ok {
				fmt.Fprintf(os.Stderr, "Unknown host: %s\n\n", arg)
				usage()
				os.Exit(1)
			}
			targets = append(targets, arg)
		}
	}

	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		log.Fatal("cannot determine script directory")
	}
	gatherDir := filepath.Dir(filename)
	hostinfoDir = filepath.Dir(gatherDir)
	goroot = gentools.DefaultGOROOT()

	if *printFlag {
		workDir = "/tmp"
	} else {
		var err error
		workDir, err = os.MkdirTemp("", "hostinfo-*")
		if err != nil {
			log.Fatal(err)
		}
		defer os.RemoveAll(workDir)
	}

	for i, target := range targets {
		if i > 0 {
			fmt.Println()
		}
		hosts[target]()
	}
}
