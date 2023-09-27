// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.regabiargs

// This file contains tests specific to making sure the register ABI
// works in a bunch of contexts in the runtime.

package runtime_test

import (
	"internal/abi"
	"internal/testenv"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"
	"time"
)

var regConfirmRun chan int

//go:registerparams
func regFinalizerPointer(v *Tint) (int, float32, [10]byte) {
	regConfirmRun <- *(*int)(v)
	return 5151, 4.0, [10]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
}

//go:registerparams
func regFinalizerIface(v Tinter) (int, float32, [10]byte) {
	regConfirmRun <- *(*int)(v.(*Tint))
	return 5151, 4.0, [10]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
}

func TestFinalizerRegisterABI(t *testing.T) {
	testenv.MustHaveExec(t)

	// Actually run the test in a subprocess because we don't want
	// finalizers from other tests interfering.
	if os.Getenv("TEST_FINALIZER_REGABI") != "1" {
		cmd := testenv.CleanCmdEnv(exec.Command(os.Args[0], "-test.run=^TestFinalizerRegisterABI$", "-test.v"))
		cmd.Env = append(cmd.Env, "TEST_FINALIZER_REGABI=1")
		out, err := cmd.CombinedOutput()
		if !strings.Contains(string(out), "PASS\n") || err != nil {
			t.Fatalf("%s\n(exit status %v)", string(out), err)
		}
		return
	}

	// Optimistically clear any latent finalizers from e.g. the testing
	// package before continuing.
	//
	// It's possible that a finalizer only becomes available to run
	// after this point, which would interfere with the test and could
	// cause a crash, but because we're running in a separate process
	// it's extremely unlikely.
	runtime.GC()
	runtime.GC()

	// fing will only pick the new IntRegArgs up if it's currently
	// sleeping and wakes up, so wait for it to go to sleep.
	success := false
	for i := 0; i < 100; i++ {
		if runtime.FinalizerGAsleep() {
			success = true
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !success {
		t.Fatal("finalizer not asleep?")
	}

	argRegsBefore := runtime.SetIntArgRegs(abi.IntArgRegs)
	defer runtime.SetIntArgRegs(argRegsBefore)

	tests := []struct {
		name         string
		fin          any
		confirmValue int
	}{
		{"Pointer", regFinalizerPointer, -1},
		{"Interface", regFinalizerIface, -2},
	}
	for i := range tests {
		test := &tests[i]
		t.Run(test.name, func(t *testing.T) {
			regConfirmRun = make(chan int)

			x := new(Tint)
			*x = (Tint)(test.confirmValue)
			runtime.SetFinalizer(x, test.fin)

			runtime.KeepAlive(x)

			// Queue the finalizer.
			runtime.GC()
			runtime.GC()

			select {
			case <-time.After(time.Second):
				t.Fatal("finalizer failed to execute")
			case gotVal := <-regConfirmRun:
				if gotVal != test.confirmValue {
					t.Fatalf("wrong finalizer executed? got %d, want %d", gotVal, test.confirmValue)
				}
			}
		})
	}
}
