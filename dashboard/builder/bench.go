// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// benchHash benchmarks a single commit.
func (b *Builder) benchHash(hash string, benchs []string) error {
	if *verbose {
		log.Println(b.name, "benchmarking", hash)
	}

	res := &PerfResult{Hash: hash, Benchmark: "meta-done"}

	// Create place in which to do work.
	workpath := filepath.Join(*buildroot, b.name+"-"+hash[:12])
	// Prepare a workpath if we don't have one we can reuse.
	update := false
	if b.lastWorkpath != workpath {
		if err := os.Mkdir(workpath, mkdirPerm); err != nil {
			return err
		}
		buildLog, _, err := b.buildRepoOnHash(workpath, hash, makeCmd)
		if err != nil {
			removePath(workpath)
			// record failure
			res.Artifacts = append(res.Artifacts, PerfArtifact{"log", buildLog})
			return b.recordPerfResult(res)
		}
		b.lastWorkpath = workpath
		update = true
	}

	// Build the benchmark binary.
	benchBin, buildLog, err := b.buildBenchmark(workpath, update)
	if err != nil {
		// record failure
		res.Artifacts = append(res.Artifacts, PerfArtifact{"log", buildLog})
		return b.recordPerfResult(res)
	}

	benchmark, procs, affinity, last := chooseBenchmark(benchBin, benchs)
	if benchmark != "" {
		res.Benchmark = fmt.Sprintf("%v-%v", benchmark, procs)
		res.Metrics, res.Artifacts, res.OK = b.executeBenchmark(workpath, hash, benchBin, benchmark, procs, affinity)
		if err = b.recordPerfResult(res); err != nil {
			return fmt.Errorf("recordResult: %s", err)
		}
	}

	if last {
		// All benchmarks have beed executed, don't need workpath anymore.
		removePath(b.lastWorkpath)
		b.lastWorkpath = ""
		// Notify the app.
		res = &PerfResult{Hash: hash, Benchmark: "meta-done", OK: true}
		if err = b.recordPerfResult(res); err != nil {
			return fmt.Errorf("recordResult: %s", err)
		}
	}

	return nil
}

// buildBenchmark builds the benchmark binary.
func (b *Builder) buildBenchmark(workpath string, update bool) (benchBin, log string, err error) {
	goroot := filepath.Join(workpath, "go")
	gobin := filepath.Join(goroot, "bin", "go") + exeExt
	gopath := filepath.Join(*buildroot, "gopath")
	env := append([]string{
		"GOROOT=" + goroot,
		"GOPATH=" + gopath},
		b.envv()...)
	// First, download without installing.
	args := []string{"get", "-d"}
	if update {
		args = append(args, "-u")
	}
	args = append(args, *benchPath)
	var buildlog bytes.Buffer
	runOpts := []runOpt{runTimeout(*buildTimeout), runEnv(env), allOutput(&buildlog), runDir(workpath)}
	err = run(exec.Command(gobin, args...), runOpts...)
	if err != nil {
		fmt.Fprintf(&buildlog, "go get -d %s failed: %s", *benchPath, err)
		return "", buildlog.String(), err
	}
	// Then, build into workpath.
	benchBin = filepath.Join(workpath, "benchbin") + exeExt
	args = []string{"build", "-o", benchBin, *benchPath}
	buildlog.Reset()
	err = run(exec.Command(gobin, args...), runOpts...)
	if err != nil {
		fmt.Fprintf(&buildlog, "go build %s failed: %s", *benchPath, err)
		return "", buildlog.String(), err
	}
	return benchBin, "", nil
}

// chooseBenchmark chooses the next benchmark to run
// based on the list of available benchmarks, already executed benchmarks
// and -benchcpu list.
func chooseBenchmark(benchBin string, doneBenchs []string) (bench string, procs, affinity int, last bool) {
	var out bytes.Buffer
	err := run(exec.Command(benchBin), allOutput(&out))
	if err != nil {
		log.Printf("Failed to query benchmark list: %v\n%s", err, out)
		last = true
		return
	}
	outStr := out.String()
	nlIdx := strings.Index(outStr, "\n")
	if nlIdx < 0 {
		log.Printf("Failed to parse benchmark list (no new line): %s", outStr)
		last = true
		return
	}
	localBenchs := strings.Split(outStr[:nlIdx], ",")
	benchsMap := make(map[string]bool)
	for _, b := range doneBenchs {
		benchsMap[b] = true
	}
	cnt := 0
	// We want to run all benchmarks with GOMAXPROCS=1 first.
	for i, procs1 := range benchCPU {
		for _, bench1 := range localBenchs {
			if benchsMap[fmt.Sprintf("%v-%v", bench1, procs1)] {
				continue
			}
			cnt++
			if cnt == 1 {
				bench = bench1
				procs = procs1
				if i < len(benchAffinity) {
					affinity = benchAffinity[i]
				}
			}
		}
	}
	last = cnt <= 1
	return
}

// executeBenchmark runs a single benchmark and parses its output.
func (b *Builder) executeBenchmark(workpath, hash, benchBin, bench string, procs, affinity int) (metrics []PerfMetric, artifacts []PerfArtifact, ok bool) {
	// Benchmarks runs mutually exclusive with other activities.
	benchMutex.RUnlock()
	defer benchMutex.RLock()
	benchMutex.Lock()
	defer benchMutex.Unlock()

	log.Printf("%v executing benchmark %v-%v on %v", b.name, bench, procs, hash)

	// The benchmark executes 'go build'/'go tool',
	// so we need properly setup env.
	env := append([]string{
		"GOROOT=" + filepath.Join(workpath, "go"),
		"PATH=" + filepath.Join(workpath, "go", "bin") + string(os.PathListSeparator) + os.Getenv("PATH"),
		"GODEBUG=gctrace=1", // since Go1.2
		"GOGCTRACE=1",       // before Go1.2
		fmt.Sprintf("GOMAXPROCS=%v", procs)},
		b.envv()...)
	args := []string{
		"-bench", bench,
		"-benchmem", strconv.Itoa(*benchMem),
		"-benchtime", benchTime.String(),
		"-benchnum", strconv.Itoa(*benchNum),
		"-tmpdir", workpath}
	if affinity != 0 {
		args = append(args, "-affinity", strconv.Itoa(affinity))
	}
	benchlog := new(bytes.Buffer)
	err := run(exec.Command(benchBin, args...), runEnv(env), allOutput(benchlog), runDir(workpath))
	if strip := benchlog.Len() - 512<<10; strip > 0 {
		// Leave the last 512K, that part contains metrics.
		benchlog = bytes.NewBuffer(benchlog.Bytes()[strip:])
	}
	artifacts = []PerfArtifact{{Type: "log", Body: benchlog.String()}}
	if err != nil {
		if err != nil {
			log.Printf("Failed to execute benchmark '%v': %v", bench, err)
			ok = false
		}
		return
	}

	metrics1, artifacts1, err := parseBenchmarkOutput(benchlog)
	if err != nil {
		log.Printf("Failed to parse benchmark output: %v", err)
		ok = false
		return
	}
	metrics = metrics1
	artifacts = append(artifacts, artifacts1...)
	ok = true
	return
}

// parseBenchmarkOutput fetches metrics and artifacts from benchmark output.
func parseBenchmarkOutput(out io.Reader) (metrics []PerfMetric, artifacts []PerfArtifact, err error) {
	s := bufio.NewScanner(out)
	metricRe := regexp.MustCompile("^GOPERF-METRIC:([a-z,0-9,-]+)=([0-9]+)$")
	fileRe := regexp.MustCompile("^GOPERF-FILE:([a-z,0-9,-]+)=(.+)$")
	for s.Scan() {
		ln := s.Text()
		if ss := metricRe.FindStringSubmatch(ln); ss != nil {
			var v uint64
			v, err = strconv.ParseUint(ss[2], 10, 64)
			if err != nil {
				err = fmt.Errorf("Failed to parse metric '%v=%v': %v", ss[1], ss[2], err)
				return
			}
			metrics = append(metrics, PerfMetric{Type: ss[1], Val: v})
		} else if ss := fileRe.FindStringSubmatch(ln); ss != nil {
			var buf []byte
			buf, err = ioutil.ReadFile(ss[2])
			if err != nil {
				err = fmt.Errorf("Failed to read file '%v': %v", ss[2], err)
				return
			}
			artifacts = append(artifacts, PerfArtifact{ss[1], string(buf)})
		}
	}
	return
}

// needsBenchmarking determines whether the commit needs benchmarking.
func needsBenchmarking(log *HgLog) bool {
	// Do not benchmark branch commits, they are usually not interesting
	// and fall out of the trunk succession.
	if log.Branch != "" {
		return false
	}
	// Do not benchmark commits that do not touch source files (e.g. CONTRIBUTORS).
	for _, f := range strings.Split(log.Files, " ") {
		if (strings.HasPrefix(f, "include") || strings.HasPrefix(f, "src")) &&
			!strings.HasSuffix(f, "_test.go") && !strings.Contains(f, "testdata") {
			return true
		}
	}
	return false
}
