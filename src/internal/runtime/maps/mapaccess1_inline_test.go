// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maps_test

import (
	"fmt"
	"internal/testenv"
	"os/exec"
	"strings"
	"testing"
)

func TestInline(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	cmd := exec.Command("go", "build", "-gcflags=-m", "internal/runtime/maps")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("exec.Command error: %v\n\n%s", err, out)
	}

	base := "runtime_mapaccess2"
	funcs := map[string]bool{
		base:              false,
		base + "_fast32":  false,
		base + "_fast64":  false,
		base + "_faststr": false,
		base + "_fat":     false,
	}

	for _, line := range strings.Split(string(out), "\n") {
		const phrase = ": inlining call to "
		if i := strings.Index(line, base); i != -1 {
			func_name := line[i:]
			fmt.Println("Found match:", func_name)
			funcs[func_name] = true // visited
		}
	}

	for f, visited := range funcs {
		if !visited {
			t.Fatalf("Didn't inline %v", f)
		}
	}
}
