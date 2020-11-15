// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"internal/testenv"
	"os/exec"
	"reflect"
	"runtime"
	"testing"
)

func TestFakeTime(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("faketime not supported on windows")
	}

	// Faketime is advanced in checkdead. External linking brings in cgo,
	// causing checkdead not working.
	testenv.MustInternalLink(t)

	t.Parallel()

	exe, err := buildTestProg(t, "testfaketime", "-tags=faketime")
	if err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	cmd := exec.Command(exe)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err = testenv.CleanCmdEnv(cmd).Run()
	if err != nil {
		t.Fatalf("exit status: %v\n%s", err, stderr.String())
	}

	t.Logf("raw stdout: %q", stdout.String())
	t.Logf("raw stderr: %q", stderr.String())

	f1, err1 := parseFakeTime(stdout.Bytes())
	if err1 != nil {
		t.Fatal(err1)
	}
	f2, err2 := parseFakeTime(stderr.Bytes())
	if err2 != nil {
		t.Fatal(err2)
	}

	const time0 = 1257894000000000000
	got := [][]fakeTimeFrame{f1, f2}
	var want = [][]fakeTimeFrame{{
		{time0 + 1, "line 2\n"},
		{time0 + 1, "line 3\n"},
		{time0 + 1e9, "line 5\n"},
		{time0 + 1e9, "2009-11-10T23:00:01Z"},
	}, {
		{time0, "line 1\n"},
		{time0 + 2, "line 4\n"},
	}}
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("want %v, got %v", want, got)
	}
}

type fakeTimeFrame struct {
	time uint64
	data string
}

func parseFakeTime(x []byte) ([]fakeTimeFrame, error) {
	var frames []fakeTimeFrame
	for len(x) != 0 {
		if len(x) < 4+8+4 {
			return nil, errors.New("truncated header")
		}
		const magic = "\x00\x00PB"
		if string(x[:len(magic)]) != magic {
			return nil, errors.New("bad magic")
		}
		x = x[len(magic):]
		time := binary.BigEndian.Uint64(x)
		x = x[8:]
		dlen := binary.BigEndian.Uint32(x)
		x = x[4:]
		data := string(x[:dlen])
		x = x[dlen:]
		frames = append(frames, fakeTimeFrame{time, data})
	}
	return frames, nil
}
