// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	. "runtime"
	"strconv"
	"strings"
	"testing"
)

var errf error

func errfn() error {
	return errf
}

func errfn1() error {
	return io.EOF
}

func BenchmarkIfaceCmp100(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < 100; j++ {
			if errfn() == io.EOF {
				b.Fatal("bad comparison")
			}
		}
	}
}

func BenchmarkIfaceCmpNil100(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < 100; j++ {
			if errfn1() == nil {
				b.Fatal("bad comparison")
			}
		}
	}
}

func BenchmarkDefer(b *testing.B) {
	for i := 0; i < b.N; i++ {
		defer1()
	}
}

func defer1() {
	defer func(x, y, z int) {
		if recover() != nil || x != 1 || y != 2 || z != 3 {
			panic("bad recover")
		}
	}(1, 2, 3)
	return
}

func BenchmarkDefer10(b *testing.B) {
	for i := 0; i < b.N/10; i++ {
		defer2()
	}
}

func defer2() {
	for i := 0; i < 10; i++ {
		defer func(x, y, z int) {
			if recover() != nil || x != 1 || y != 2 || z != 3 {
				panic("bad recover")
			}
		}(1, 2, 3)
	}
}

func BenchmarkDeferMany(b *testing.B) {
	for i := 0; i < b.N; i++ {
		defer func(x, y, z int) {
			if recover() != nil || x != 1 || y != 2 || z != 3 {
				panic("bad recover")
			}
		}(1, 2, 3)
	}
}

// The profiling signal handler needs to know whether it is executing runtime.gogo.
// The constant RuntimeGogoBytes in arch_*.h gives the size of the function;
// we don't have a way to obtain it from the linker (perhaps someday).
// Test that the constant matches the size determined by 'go tool nm -S'.
// The value reported will include the padding between runtime.gogo and the
// next function in memory. That's fine.
func TestRuntimeGogoBytes(t *testing.T) {
	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	out, err := exec.Command("go", "build", "-o", dir+"/hello", "../../../test/helloworld.go").CombinedOutput()
	if err != nil {
		t.Fatalf("building hello world: %v\n%s", err, out)
	}

	out, err = exec.Command("go", "tool", "nm", "-size", dir+"/hello").CombinedOutput()
	if err != nil {
		t.Fatalf("go tool nm: %v\n%s", err, out)
	}

	for _, line := range strings.Split(string(out), "\n") {
		f := strings.Fields(line)
		if len(f) == 4 && f[3] == "runtime.gogo" {
			size, _ := strconv.Atoi(f[1])
			if GogoBytes() != int32(size) {
				t.Fatalf("RuntimeGogoBytes = %d, should be %d", GogoBytes(), size)
			}
			return
		}
	}

	t.Fatalf("go tool nm did not report size for runtime.gogo")
}
