// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pods_test

import (
	"fmt"
	"hash/fnv"
	"internal/coverage"
	"internal/coverage/pods"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestPodCollection(t *testing.T) {
	//testenv.MustHaveGoBuild(t)

	mkdir := func(d string, perm os.FileMode) string {
		dp := filepath.Join(t.TempDir(), d)
		if err := os.Mkdir(dp, perm); err != nil {
			t.Fatal(err)
		}
		return dp
	}

	mkfile := func(d string, fn string) string {
		fp := filepath.Join(d, fn)
		if err := os.WriteFile(fp, []byte("foo"), 0666); err != nil {
			t.Fatal(err)
		}
		return fp
	}

	mkmeta := func(dir string, tag string) string {
		h := fnv.New128a()
		h.Write([]byte(tag))
		hash := h.Sum(nil)
		fn := fmt.Sprintf("%s.%x", coverage.MetaFilePref, hash)
		return mkfile(dir, fn)
	}

	mkcounter := func(dir string, tag string, nt int, pid int) string {
		h := fnv.New128a()
		h.Write([]byte(tag))
		hash := h.Sum(nil)
		fn := fmt.Sprintf(coverage.CounterFileTempl, coverage.CounterFilePref, hash, pid, nt)
		return mkfile(dir, fn)
	}

	trim := func(path string) string {
		b := filepath.Base(path)
		d := filepath.Dir(path)
		db := filepath.Base(d)
		return db + "/" + b
	}

	podToString := func(p pods.Pod) string {
		rv := trim(p.MetaFile) + " [\n"
		for k, df := range p.CounterDataFiles {
			rv += trim(df)
			if p.Origins != nil {
				rv += fmt.Sprintf(" o:%d", p.Origins[k])
			}
			rv += "\n"
		}
		return rv + "]"
	}

	// Create a couple of directories.
	o1 := mkdir("o1", 0777)
	o2 := mkdir("o2", 0777)

	// Add some random files (not coverage related)
	mkfile(o1, "blah.txt")
	mkfile(o1, "something.exe")

	// Add a meta-data file with two counter files to first dir.
	mkmeta(o1, "m1")
	mkcounter(o1, "m1", 1, 42)
	mkcounter(o1, "m1", 2, 41)
	mkcounter(o1, "m1", 2, 40)

	// Add a counter file with no associated meta file.
	mkcounter(o1, "orphan", 9, 39)

	// Add a meta-data file with three counter files to second dir.
	mkmeta(o2, "m2")
	mkcounter(o2, "m2", 1, 38)
	mkcounter(o2, "m2", 2, 37)
	mkcounter(o2, "m2", 3, 36)

	// Add a duplicate of the first meta-file and a corresponding
	// counter file to the second dir. This is intended to capture
	// the scenario where we have two different runs of the same
	// coverage-instrumented binary, but with the output files
	// sent to separate directories.
	mkmeta(o2, "m1")
	mkcounter(o2, "m1", 11, 35)

	// Collect pods.
	podlist, err := pods.CollectPods([]string{o1, o2}, true)
	if err != nil {
		t.Fatal(err)
	}

	// Verify pods
	if len(podlist) != 2 {
		t.Fatalf("expected 2 pods got %d pods", len(podlist))
	}

	for k, p := range podlist {
		t.Logf("%d: mf=%s\n", k, p.MetaFile)
	}

	expected := []string{
		`o1/covmeta.0880952782ab1be95aa0733055a4d06b [
o1/covcounters.0880952782ab1be95aa0733055a4d06b.40.2 o:0
o1/covcounters.0880952782ab1be95aa0733055a4d06b.41.2 o:0
o1/covcounters.0880952782ab1be95aa0733055a4d06b.42.1 o:0
o2/covcounters.0880952782ab1be95aa0733055a4d06b.35.11 o:1
]`,
		`o2/covmeta.0880952783ab1be95aa0733055a4d1a6 [
o2/covcounters.0880952783ab1be95aa0733055a4d1a6.36.3 o:1
o2/covcounters.0880952783ab1be95aa0733055a4d1a6.37.2 o:1
o2/covcounters.0880952783ab1be95aa0733055a4d1a6.38.1 o:1
]`,
	}
	for k, exp := range expected {
		got := podToString(podlist[k])
		if exp != got {
			t.Errorf("pod %d: expected:\n%s\ngot:\n%s", k, exp, got)
		}
	}

	// Check handling of bad/unreadable dir.
	if runtime.GOOS == "linux" {
		dbad := "/dev/null"
		_, err = pods.CollectPods([]string{dbad}, true)
		if err == nil {
			t.Errorf("executed error due to unreadable dir")
		}
	}
}
