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
	"time"
)

func TestFakeTime(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("faketime not supported on windows")
	}

	// Faketime is advanced in checkdead. External linking brings in cgo,
	// causing checkdead not working.
	testenv.MustInternalLink(t, deadlockBuildTypes)

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

func TestTimeTimerType(t *testing.T) {
	// runtime.timeTimer (exported for testing as TimeTimer)
	// must have time.Timer and time.Ticker as a prefix
	// (meaning those two must have the same layout).
	runtimeTimeTimer := reflect.TypeOf(runtime.TimeTimer{})

	check := func(name string, typ reflect.Type) {
		n1 := runtimeTimeTimer.NumField()
		n2 := typ.NumField()
		if n1 != n2+1 {
			t.Errorf("runtime.TimeTimer has %d fields, want %d (%s has %d fields)", n1, n2+1, name, n2)
			return
		}
		for i := 0; i < n2; i++ {
			f1 := runtimeTimeTimer.Field(i)
			f2 := typ.Field(i)
			t1 := f1.Type
			t2 := f2.Type
			if t1 != t2 && !(t1.Kind() == reflect.UnsafePointer && t2.Kind() == reflect.Chan) {
				t.Errorf("runtime.Timer field %s %v incompatible with %s field %s %v", f1.Name, t1, name, f2.Name, t2)
			}
			if f1.Offset != f2.Offset {
				t.Errorf("runtime.Timer field %s offset %d incompatible with %s field %s offset %d", f1.Name, f1.Offset, name, f2.Name, f2.Offset)
			}
		}
	}

	check("time.Timer", reflect.TypeOf(time.Timer{}))
	check("time.Ticker", reflect.TypeOf(time.Ticker{}))
}
