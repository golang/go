// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing_test

import (
	"flag"
	"internal/testenv"
	"os"
	"os/exec"
	"testing"
)

var testFlagArg = flag.String("test_flag_arg", "", "TestFlag: passing -v option")

const flagTestEnv = "GO_WANT_FLAG_HELPER_PROCESS"

func TestFlag(t *testing.T) {
	if os.Getenv(flagTestEnv) == "1" {
		testFlagHelper(t)
		return
	}

	testenv.MustHaveExec(t)

	for _, flag := range []string{"", "-test.v", "-test.v=test2json"} {
		flag := flag
		t.Run(flag, func(t *testing.T) {
			t.Parallel()
			exe, err := os.Executable()
			if err != nil {
				exe = os.Args[0]
			}
			cmd := exec.Command(exe, "-test.run=^TestFlag$", "-test_flag_arg="+flag)
			if flag != "" {
				cmd.Args = append(cmd.Args, flag)
			}
			cmd.Env = append(cmd.Environ(), flagTestEnv+"=1")
			b, err := cmd.CombinedOutput()
			if len(b) > 0 {
				// When we set -test.v=test2json, we need to escape the ^V control
				// character used for JSON framing so that the JSON parser doesn't
				// misinterpret the subprocess output as output from the parent test.
				t.Logf("%q", b)
			}
			if err != nil {
				t.Error(err)
			}
		})
	}
}

// testFlagHelper is called by the TestFlagHelper subprocess.
func testFlagHelper(t *testing.T) {
	f := flag.Lookup("test.v")
	if f == nil {
		t.Fatal(`flag.Lookup("test.v") failed`)
	}

	bf, ok := f.Value.(interface{ IsBoolFlag() bool })
	if !ok {
		t.Errorf("test.v flag (type %T) does not have IsBoolFlag method", f)
	} else if !bf.IsBoolFlag() {
		t.Error("test.v IsBoolFlag() returned false")
	}

	gf, ok := f.Value.(flag.Getter)
	if !ok {
		t.Fatalf("test.v flag (type %T) does not have Get method", f)
	}
	v := gf.Get()

	var want any
	switch *testFlagArg {
	case "":
		want = false
	case "-test.v":
		want = true
	case "-test.v=test2json":
		want = "test2json"
	default:
		t.Fatalf("unexpected test_flag_arg %q", *testFlagArg)
	}

	if v != want {
		t.Errorf("test.v is %v want %v", v, want)
	}
}
