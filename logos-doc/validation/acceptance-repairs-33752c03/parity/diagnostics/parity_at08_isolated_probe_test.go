package goparity

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

func TestParityAt08IsolatedProbe(t *testing.T) {
	const program = "at08_recursion_bound"
	inputs := []struct {
		path string
		sum  string
	}{
		{"parity_test.go", "1be848e4e5f94f76d3484e12b253d4eaad81990a89635a59cf59f86fa1149bb4"},
		{filepath.Join("testdata", "run", program, "main.go"), "e1aa8d47d3fe4ddc0d43be98ec361c1ad2dcf9ca8711f1feaa49afc2081bc359"},
	}
	for _, input := range inputs {
		if got := parityAt08ProbeHash(t, input.path); got != input.sum {
			t.Fatalf("probe input %s hash = %s, want %s", input.path, got, input.sum)
		}
		t.Logf("probe input path=%s sha256=%s", input.path, input.sum)
	}

	fork := forkTool(t)
	up, upOn := upstreamTool(t)
	if !upOn {
		t.Fatal("the isolated probe requires the pinned upstream oracle")
	}
	var k6 config
	for _, c := range forkConfigs {
		if c.id == "K6" {
			k6 = c
			break
		}
	}
	var u3n upConfig
	for _, u := range upConfigs {
		if u.id == "U3N" {
			u3n = u
			break
		}
	}
	if k6.id == "" || u3n.id == "" {
		t.Fatal("the frozen K6 or U3N configuration is absent")
	}

	programs := []string{program}
	base := t.TempDir()
	start := time.Now()
	forkBuild := buildFork(fork, base, programs, k6)
	t.Logf("probe build id=K6 elapsed=%s error=%v", time.Since(start), forkBuild.err)
	if forkBuild.err != nil {
		t.Fatalf("K6 build: %v\n%s", forkBuild.err, forkBuild.output)
	}
	start = time.Now()
	upBuild := buildUpstream(up, base, programs, u3n)
	t.Logf("probe build id=U3N elapsed=%s error=%v", time.Since(start), upBuild.err)
	if upBuild.err != nil {
		t.Fatalf("U3N build: %v\n%s", upBuild.err, upBuild.output)
	}
	rows := []struct {
		id    string
		args  []string
		build buildResult
		sum   string
	}{
		{id: "K6", args: k6.args, build: forkBuild},
		{id: "U3N", args: u3n.args, build: upBuild},
	}
	for i := range rows {
		row := &rows[i]
		if !slices.Equal(row.build.progs, programs) {
			t.Fatalf("%s built %v, want %v", row.id, row.build.progs, programs)
		}
		path := filepath.Join(row.build.outDir, program)
		row.sum = parityAt08ProbeHash(t, path)
		t.Logf("probe binary id=%s path=%s sha256=%s args=%q", row.id, path, row.sum, row.args)
	}

	want := observation{stdout: "ok\n"}
	for trial := 1; trial <= 3; trial++ {
		for _, row := range rows {
			path := filepath.Join(row.build.outDir, program)
			start := time.Now()
			obs, err := runBinary(path, "4", row.args)
			t.Logf("probe run id=%s trial=%d budget=10s GOMAXPROCS=4 elapsed=%s error=%v exit=%d stdout=%q stderr=%q",
				row.id, trial, time.Since(start), err, obs.exit, obs.stdout, obs.stderr)
			if err != nil {
				t.Errorf("%s trial %d: %v", row.id, trial, err)
				continue
			}
			if corpusDefect(obs) || !obsEqual(obs, want) {
				t.Errorf("%s trial %d: got %s, want exit 0, stdout ok, and empty stderr", row.id, trial, obsDesc(obs))
			}
		}
	}
	for _, row := range rows {
		if got := parityAt08ProbeHash(t, filepath.Join(row.build.outDir, program)); got != row.sum {
			t.Errorf("%s binary changed: got %s, want %s", row.id, got, row.sum)
		}
	}
	for _, input := range inputs {
		if got := parityAt08ProbeHash(t, input.path); got != input.sum {
			t.Errorf("probe input %s changed: got %s, want %s", input.path, got, input.sum)
		}
	}
}

func parityAt08ProbeHash(t *testing.T, path string) string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		t.Fatal(err)
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}
