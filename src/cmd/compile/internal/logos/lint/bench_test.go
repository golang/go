// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// benchFuncs is the number of generated functions BenchmarkLintOverhead
// compiles.
const benchFuncs = 400

// benchSource returns a loGo/s package of n functions with loops, branches,
// switches, error results, and calls between them, and no finding.
func benchSource(n int) string {
	var s strings.Builder
	s.WriteString("package p\n\ntype E string\n\nfunc (e E) Error() string { return string(e) }\n\n")
	for i := range n {
		fmt.Fprintf(&s, `func f%[1]d(a, b int) (int, error) {
	total := 0
	for i := 0; i < a; i++ {
		if i%%2 == 0 {
			total += i * b
		} else {
			total -= i
		}
	}
	switch {
	case total > 100:
		total /= 2
	case total < -100:
		total *= 2
	}
	if total < 0 {
		return 0, E("negative")
	}
`, i)
		if i > 0 {
			fmt.Fprintf(&s, "\tr, err := f%d(total, 1)\n\tif err != nil {\n\t\treturn 0, err\n\t}\n\ttotal += r\n", i-1)
		}
		s.WriteString("\treturn total, nil\n}\n\n")
	}
	return s.String()
}

// compiler runs the compile tool on one source file and returns the CPU time
// of the process, user plus system.
type compiler struct {
	b      *testing.B
	tool   string
	object string
	source string
}

// run compiles the source with the extra flags and returns the CPU time the
// compile used. Wall time on a loaded host is mostly scheduling noise, and a
// single backend worker (-c=1) keeps the CPU time of equal compiles equal.
func (c compiler) run(extra ...string) time.Duration {
	args := []string{"-p", "p", "-c=1", "-o", c.object}
	args = append(append(args, extra...), c.source)
	cmd := exec.Command(c.tool, args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		c.b.Fatalf("compile: %v\n%s", err, out)
	}
	return cmd.ProcessState.UserTime() + cmd.ProcessState.SystemTime()
}

// percentiles reports the median, 95th percentile, and maximum of the
// ratios as metrics named prefix-p50-%, prefix-p95-%, and prefix-max-%.
func percentiles(b *testing.B, prefix string, ratios []float64) {
	slices.Sort(ratios)
	b.ReportMetric(ratios[len(ratios)/2], prefix+"-p50-%")
	b.ReportMetric(ratios[len(ratios)*95/100], prefix+"-p95-%")
	b.ReportMetric(ratios[len(ratios)-1], prefix+"-max-%")
}

// BenchmarkLintOverhead measures the CPU cost of the lints inside a whole
// compile (SPEC Sec 1.2: target 2% of compile time). Each iteration compiles
// a generated loGo/s package three times: without lints, with -logos.cfg,
// and without lints again. The lint-* metrics are the extra CPU of the second
// compile over the first, in percent; the null-* metrics are the same ratio
// for the third over the first, so they measure the noise of the host, not
// the lints. The default policy runs every registered rule at its default
// severity. The package is generated and has no finding: only the rules that
// exist today run, so the number is a floor.
func BenchmarkLintOverhead(b *testing.B) {
	testenv.MustHaveGoBuild(b)
	dir := b.TempDir()
	c := compiler{b: b, object: filepath.Join(dir, "out.a"), source: filepath.Join(dir, "p.lgo")}
	policy := filepath.Join(dir, "policy.json")
	for path, data := range map[string]string{c.source: benchSource(benchFuncs), policy: `{"version":1}`} {
		if err := os.WriteFile(path, []byte(data), 0o666); err != nil {
			b.Fatal(err)
		}
	}
	toolDir, err := exec.Command(testenv.GoToolPath(b), "env", "GOTOOLDIR").Output()
	if err != nil {
		b.Fatal(err)
	}
	c.tool = filepath.Join(strings.TrimSpace(string(toolDir)), "compile")
	c.run()
	c.run("-logos.cfg=" + policy)
	b.ResetTimer()
	lint := make([]float64, 0, b.N)
	null := make([]float64, 0, b.N)
	for range b.N {
		plain := c.run()
		linted := c.run("-logos.cfg=" + policy)
		again := c.run()
		lint = append(lint, float64(linted-plain)/float64(plain)*100)
		null = append(null, float64(again-plain)/float64(plain)*100)
	}
	percentiles(b, "lint", lint)
	percentiles(b, "null", null)
}

// BenchmarkLintRun measures one Run of every registered rule over the
// type-checked source of discardedSrc, in process.
func BenchmarkLintRun(b *testing.B) {
	testenv.MustHaveGoBuild(b)
	in := check(b, KindLgo, "x.lgo", discardedSrc)
	b.ResetTimer()
	for range b.N {
		Run(in)
	}
}
