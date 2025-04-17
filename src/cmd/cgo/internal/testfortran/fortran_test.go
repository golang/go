// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fortran

import (
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestFortran(t *testing.T) {
	testenv.MustHaveGoRun(t)
	testenv.MustHaveCGO(t)

	// Find the FORTRAN compiler.
	fc := os.Getenv("FC")
	if fc == "" {
		fc, _ = exec.LookPath("gfortran")
	}
	if fc == "" {
		t.Skip("fortran compiler not found (try setting $FC)")
	}

	var fcExtra []string
	if strings.Contains(fc, "gfortran") {
		// TODO: This duplicates but also diverges from logic from cmd/go
		// itself. For example, cmd/go merely adds -lgfortran without the extra
		// library path work. If this is what's necessary to run gfortran, we
		// should reconcile the logic here and in cmd/go.. Maybe this should
		// become a cmd/go script test to share that logic.

		// Add -m32 if we're targeting 386, in case this is a cross-compile.
		if runtime.GOARCH == "386" {
			fcExtra = append(fcExtra, "-m32")
		}

		// Find libgfortran. If the FORTRAN compiler isn't bundled
		// with the C linker, this may be in a path the C linker can't
		// find on its own. (See #14544)
		libExt := "so"
		switch runtime.GOOS {
		case "darwin":
			libExt = "dylib"
		case "aix":
			libExt = "a"
		}
		libPath, err := exec.Command(fc, append([]string{"-print-file-name=libgfortran." + libExt}, fcExtra...)...).CombinedOutput()
		if err != nil {
			t.Errorf("error invoking %s: %s", fc, err)
		}
		libDir := filepath.Dir(string(libPath))
		cgoLDFlags := os.Getenv("CGO_LDFLAGS")
		cgoLDFlags += " -L " + libDir
		if runtime.GOOS != "aix" {
			cgoLDFlags += " -Wl,-rpath," + libDir
		}
		t.Logf("CGO_LDFLAGS=%s", cgoLDFlags)
		os.Setenv("CGO_LDFLAGS", cgoLDFlags)

	}

	// Do a test build that doesn't involve Go FORTRAN support.
	fcArgs := append([]string{"testdata/helloworld/helloworld.f90", "-o", "/dev/null"}, fcExtra...)
	t.Logf("%s %s", fc, fcArgs)
	if err := exec.Command(fc, fcArgs...).Run(); err != nil {
		t.Skipf("skipping Fortran test: could not build helloworld.f90 with %s: %s", fc, err)
	}

	// Finally, run the actual test.
	t.Log("go", "run", "./testdata/testprog")
	var stdout, stderr strings.Builder
	cmd := exec.Command(testenv.GoToolPath(t), "run", "./testdata/testprog")
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	t.Logf("%v", cmd)
	if stderr.Len() != 0 {
		t.Logf("stderr:\n%s", stderr.String())
	}
	if err != nil {
		t.Errorf("%v\n%s", err, stdout.String())
	} else if stdout.String() != "ok\n" {
		t.Errorf("stdout:\n%s\nwant \"ok\"", stdout.String())
	}
}
