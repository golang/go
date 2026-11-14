// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"fmt"
	"internal/cpu"
	"internal/testenv"
	"os"
	"os/exec"
	"runtime"
	"simd/archsimd"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func TestAsyncPreemptSIMD(t *testing.T) {
	if runtime.GOOS == "plan9" {
		t.Skip("asynchronous preemption not supported on plan9")
	}
	if strings.Contains(os.Getenv("GODEBUG"), "asyncpreemptoff=1") {
		t.Skip("asynchronous preemption disabled via GODEBUG")
	}
	if os.Getenv("GO_TEST_PREEMPT_XGETBV1_OFF") == "1" && runtime.GOOS != "windows" && cpu.X86.HasXGETBV1 {
		t.Fatalf("expected cpu.X86.HasXGETBV1 == false under GODEBUG=cpu.xgetbv1=off")
	}

	t.Run("Y", testAsyncPreemptY)
	t.Run("Z", testAsyncPreemptZ)
	t.Run("XHiWithMask", testAsyncPreemptXHiWithMask)
	t.Run("Mask", testAsyncPreemptMask)
	t.Run("ZAndMask", testAsyncPreemptZAndMask)
	t.Run("MixedCleanDirty", testAsyncPreemptMixedCleanDirty)

	if os.Getenv("GO_TEST_PREEMPT_XGETBV1_OFF") == "" {
		t.Run("xgetbv1_off", func(t *testing.T) {
			testenv.MustHaveExec(t)
			cmd := exec.Command(testenv.Executable(t), "-test.run=^TestAsyncPreemptSIMD$")
			if testing.Short() {
				cmd.Args = append(cmd.Args, "-test.short")
			}
			env := os.Environ()
			godebug := os.Getenv("GODEBUG")
			if godebug != "" {
				godebug += ",cpu.xgetbv1=off"
			} else {
				godebug = "cpu.xgetbv1=off"
			}
			cmd.Env = append(env, "GO_TEST_PREEMPT_XGETBV1_OFF=1", "GODEBUG="+godebug)
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("TestAsyncPreemptSIMD with GODEBUG=%s failed: %v\n%s", godebug, err, out)
			}
		})
	}
}

type preemptWorker struct {
	progress atomic.Uint64
	run      func(stop *atomic.Uint32, progress *atomic.Uint64) error
}

func runPreemptWorkers(t *testing.T, workers []*preemptWorker) {
	t.Helper()
	// Use 2 Ps so multiple workers share Ms and overwrite each other's vector
	// and opmask registers while parked, exercising both save/restore and
	// clean-to-dirty zeroing transitions.
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))

	var stop atomic.Uint32
	errs := make([]error, len(workers))
	var wg sync.WaitGroup
	for i, w := range workers {
		wg.Add(1)
		go func(i int, w *preemptWorker) {
			defer wg.Done()
			errs[i] = w.run(&stop, &w.progress)
		}(i, w)
	}

	numGCs := 8
	if testing.Short() {
		numGCs = 4
	}
	for range numGCs {
		// Wait until every worker has advanced since the previous GC so we
		// exercise multiple distinct save/restore cycles per worker.
		for _, w := range workers {
			prev := w.progress.Load()
			for w.progress.Load() == prev {
				runtime.Gosched()
			}
		}
		runtime.GC()
	}

	stop.Store(1)
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("worker %d: %v", i, err)
		}
	}
}

func testAsyncPreemptY(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("requires AVX2")
	}
	runPreemptWorkers(t, []*preemptWorker{
		{run: func(stop *atomic.Uint32, progress *atomic.Uint64) error {
			return runYLoop(stop, progress, 0x1111_0000_0000_0001)
		}},
		{run: func(stop *atomic.Uint32, progress *atomic.Uint64) error {
			return runYLoop(stop, progress, 0x2222_0000_0000_0003)
		}},
	})
}

func testAsyncPreemptZ(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("requires AVX-512")
	}
	runPreemptWorkers(t, []*preemptWorker{
		{run: func(stop *atomic.Uint32, progress *atomic.Uint64) error {
			return runZLoop(stop, progress, 0x3333_0000_0000_0001)
		}},
		{run: func(stop *atomic.Uint32, progress *atomic.Uint64) error {
			return runZLoop(stop, progress, 0x4444_0000_0000_0005)
		}},
	})
}

func testAsyncPreemptXHiWithMask(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("requires AVX-512")
	}
	runPreemptWorkers(t, []*preemptWorker{
		{run: func(stop *atomic.Uint32, progress *atomic.Uint64) error {
			return runX28MaskedLoop(stop, progress, 0x5555_0000_0000_0001, 0b11)
		}},
		{run: func(stop *atomic.Uint32, progress *atomic.Uint64) error {
			return runX28MaskedLoop(stop, progress, 0x6666_0000_0000_0007, 0b01)
		}},
	})
}

func testAsyncPreemptMask(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("requires AVX-512")
	}
	runPreemptWorkers(t, []*preemptWorker{
		{run: func(stop *atomic.Uint32, progress *atomic.Uint64) error {
			return runMaskLoop(stop, progress, 0x1234_5678_9abc_def0)
		}},
		{run: func(stop *atomic.Uint32, progress *atomic.Uint64) error {
			return runMaskLoop(stop, progress, 0xfedc_ba98_7654_3210)
		}},
	})
}

func testAsyncPreemptZAndMask(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("requires AVX-512")
	}
	runPreemptWorkers(t, []*preemptWorker{
		{run: func(stop *atomic.Uint32, progress *atomic.Uint64) error {
			return runZAndMaskLoop(stop, progress, 0x7777_0000_0000_0001, 0xaa55_cc33_f00f_9669)
		}},
		{run: func(stop *atomic.Uint32, progress *atomic.Uint64) error {
			return runZAndMaskLoop(stop, progress, 0x8888_0000_0000_0009, 0x55aa_33cc_0ff0_6996)
		}},
	})
}

func testAsyncPreemptMixedCleanDirty(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("requires AVX2")
	}
	workers := []*preemptWorker{
		{run: func(stop *atomic.Uint32, progress *atomic.Uint64) error {
			return runX12Loop(stop, progress, 0x9999_0000_0000_0001)
		}},
		{run: func(stop *atomic.Uint32, progress *atomic.Uint64) error {
			return runYLoop(stop, progress, 0xaaaa_0000_0000_0003)
		}},
	}
	if archsimd.X86.AVX512() {
		workers = append(workers,
			&preemptWorker{run: func(stop *atomic.Uint32, progress *atomic.Uint64) error {
				return runZAndMaskLoop(stop, progress, 0xbbbb_0000_0000_0005, 0xc3c3_a5a5_5a5a_3c3c)
			}},
		)
	}
	runPreemptWorkers(t, workers)
}

func makeInit2(seed uint64, n int) [][2]int64 {
	inits := make([][2]int64, n)
	for i := range n {
		for j := range 2 {
			inits[i][j] = int64(seed) + int64(i*2+j)*0x1000_0001 + int64(j+1)<<40
		}
	}
	return inits
}

func makeInit4(seed uint64, n int) [][4]int64 {
	inits := make([][4]int64, n)
	for i := range n {
		for j := range 4 {
			inits[i][j] = int64(seed) + int64(i*4+j)*0x1000_0001 + int64(j+1)<<40
		}
	}
	return inits
}

func makeInit8(seed uint64, n int) [][8]int64 {
	inits := make([][8]int64, n)
	for i := range n {
		for j := range 8 {
			inits[i][j] = int64(seed) + int64(i*8+j)*0x1000_0001 + int64(j+1)<<40
		}
	}
	return inits
}

// runX12Loop keeps 13 Int64x2 values (12 accumulators + delta) live in X0-X14
// with no YMM/ZMM/K instructions.
//
//go:noinline
func runX12Loop(stop *atomic.Uint32, progress *atomic.Uint64, seed uint64) error {
	inits := makeInit2(seed, 12)
	deltaArr := [2]int64{1, 3}
	d := archsimd.LoadInt64x2Array(&deltaArr)

	v0 := archsimd.LoadInt64x2Array(&inits[0])
	v1 := archsimd.LoadInt64x2Array(&inits[1])
	v2 := archsimd.LoadInt64x2Array(&inits[2])
	v3 := archsimd.LoadInt64x2Array(&inits[3])
	v4 := archsimd.LoadInt64x2Array(&inits[4])
	v5 := archsimd.LoadInt64x2Array(&inits[5])
	v6 := archsimd.LoadInt64x2Array(&inits[6])
	v7 := archsimd.LoadInt64x2Array(&inits[7])
	v8 := archsimd.LoadInt64x2Array(&inits[8])
	v9 := archsimd.LoadInt64x2Array(&inits[9])
	v10 := archsimd.LoadInt64x2Array(&inits[10])
	v11 := archsimd.LoadInt64x2Array(&inits[11])

	var iters int64
	for stop.Load() == 0 {
		iters++
		v0 = v0.Add(d)
		v1 = v1.Add(d)
		v2 = v2.Add(d)
		v3 = v3.Add(d)
		v4 = v4.Add(d)
		v5 = v5.Add(d)
		v6 = v6.Add(d)
		v7 = v7.Add(d)
		v8 = v8.Add(d)
		v9 = v9.Add(d)
		v10 = v10.Add(d)
		v11 = v11.Add(d)
		if iters&1023 == 0 {
			progress.Store(uint64(iters))
		}
	}

	var got [12][2]int64
	v0.StoreArray(&got[0])
	v1.StoreArray(&got[1])
	v2.StoreArray(&got[2])
	v3.StoreArray(&got[3])
	v4.StoreArray(&got[4])
	v5.StoreArray(&got[5])
	v6.StoreArray(&got[6])
	v7.StoreArray(&got[7])
	v8.StoreArray(&got[8])
	v9.StoreArray(&got[9])
	v10.StoreArray(&got[10])
	v11.StoreArray(&got[11])

	for i := range got {
		for j := range 2 {
			want := inits[i][j] + iters*deltaArr[j]
			if got[i][j] != want {
				return fmt.Errorf("X12 reg %d lane %d after %d iters: got %#x, want %#x", i, j, iters, got[i][j], want)
			}
		}
	}
	return nil
}

// runYLoop keeps 14 Int64x4 values (13 accumulators + delta) live in Y0-Y14.
//
//go:noinline
func runYLoop(stop *atomic.Uint32, progress *atomic.Uint64, seed uint64) error {
	inits := makeInit4(seed, 13)
	deltaArr := [4]int64{1, 3, 5, 7}
	d := archsimd.LoadInt64x4Array(&deltaArr)

	v0 := archsimd.LoadInt64x4Array(&inits[0])
	v1 := archsimd.LoadInt64x4Array(&inits[1])
	v2 := archsimd.LoadInt64x4Array(&inits[2])
	v3 := archsimd.LoadInt64x4Array(&inits[3])
	v4 := archsimd.LoadInt64x4Array(&inits[4])
	v5 := archsimd.LoadInt64x4Array(&inits[5])
	v6 := archsimd.LoadInt64x4Array(&inits[6])
	v7 := archsimd.LoadInt64x4Array(&inits[7])
	v8 := archsimd.LoadInt64x4Array(&inits[8])
	v9 := archsimd.LoadInt64x4Array(&inits[9])
	v10 := archsimd.LoadInt64x4Array(&inits[10])
	v11 := archsimd.LoadInt64x4Array(&inits[11])
	v12 := archsimd.LoadInt64x4Array(&inits[12])

	var iters int64
	for stop.Load() == 0 {
		iters++
		v0 = v0.Add(d)
		v1 = v1.Add(d)
		v2 = v2.Add(d)
		v3 = v3.Add(d)
		v4 = v4.Add(d)
		v5 = v5.Add(d)
		v6 = v6.Add(d)
		v7 = v7.Add(d)
		v8 = v8.Add(d)
		v9 = v9.Add(d)
		v10 = v10.Add(d)
		v11 = v11.Add(d)
		v12 = v12.Add(d)
		if iters&1023 == 0 {
			progress.Store(uint64(iters))
		}
	}

	var got [13][4]int64
	v0.StoreArray(&got[0])
	v1.StoreArray(&got[1])
	v2.StoreArray(&got[2])
	v3.StoreArray(&got[3])
	v4.StoreArray(&got[4])
	v5.StoreArray(&got[5])
	v6.StoreArray(&got[6])
	v7.StoreArray(&got[7])
	v8.StoreArray(&got[8])
	v9.StoreArray(&got[9])
	v10.StoreArray(&got[10])
	v11.StoreArray(&got[11])
	v12.StoreArray(&got[12])

	for i := range got {
		for j := range 4 {
			want := inits[i][j] + iters*deltaArr[j]
			if got[i][j] != want {
				return fmt.Errorf("Y reg %d lane %d after %d iters: got %#x, want %#x", i, j, iters, got[i][j], want)
			}
		}
	}
	return nil
}

// runZLoop keeps 31 Int64x8 values (30 accumulators + delta) live across
// all allocatable Z registers (Z0-Z14 and Z16-Z31).
//
//go:noinline
func runZLoop(stop *atomic.Uint32, progress *atomic.Uint64, seed uint64) error {
	if !archsimd.X86.AVX512() {
		return nil
	}
	inits := makeInit8(seed, 30)
	deltaArr := [8]int64{1, 3, 5, 7, 11, 13, 17, 19}
	d := archsimd.LoadInt64x8Array(&deltaArr)

	v0 := archsimd.LoadInt64x8Array(&inits[0])
	v1 := archsimd.LoadInt64x8Array(&inits[1])
	v2 := archsimd.LoadInt64x8Array(&inits[2])
	v3 := archsimd.LoadInt64x8Array(&inits[3])
	v4 := archsimd.LoadInt64x8Array(&inits[4])
	v5 := archsimd.LoadInt64x8Array(&inits[5])
	v6 := archsimd.LoadInt64x8Array(&inits[6])
	v7 := archsimd.LoadInt64x8Array(&inits[7])
	v8 := archsimd.LoadInt64x8Array(&inits[8])
	v9 := archsimd.LoadInt64x8Array(&inits[9])
	v10 := archsimd.LoadInt64x8Array(&inits[10])
	v11 := archsimd.LoadInt64x8Array(&inits[11])
	v12 := archsimd.LoadInt64x8Array(&inits[12])
	v13 := archsimd.LoadInt64x8Array(&inits[13])
	v14 := archsimd.LoadInt64x8Array(&inits[14])
	v15 := archsimd.LoadInt64x8Array(&inits[15])
	v16 := archsimd.LoadInt64x8Array(&inits[16])
	v17 := archsimd.LoadInt64x8Array(&inits[17])
	v18 := archsimd.LoadInt64x8Array(&inits[18])
	v19 := archsimd.LoadInt64x8Array(&inits[19])
	v20 := archsimd.LoadInt64x8Array(&inits[20])
	v21 := archsimd.LoadInt64x8Array(&inits[21])
	v22 := archsimd.LoadInt64x8Array(&inits[22])
	v23 := archsimd.LoadInt64x8Array(&inits[23])
	v24 := archsimd.LoadInt64x8Array(&inits[24])
	v25 := archsimd.LoadInt64x8Array(&inits[25])
	v26 := archsimd.LoadInt64x8Array(&inits[26])
	v27 := archsimd.LoadInt64x8Array(&inits[27])
	v28 := archsimd.LoadInt64x8Array(&inits[28])
	v29 := archsimd.LoadInt64x8Array(&inits[29])

	var iters int64
	for stop.Load() == 0 {
		iters++
		v0 = v0.Add(d)
		v1 = v1.Add(d)
		v2 = v2.Add(d)
		v3 = v3.Add(d)
		v4 = v4.Add(d)
		v5 = v5.Add(d)
		v6 = v6.Add(d)
		v7 = v7.Add(d)
		v8 = v8.Add(d)
		v9 = v9.Add(d)
		v10 = v10.Add(d)
		v11 = v11.Add(d)
		v12 = v12.Add(d)
		v13 = v13.Add(d)
		v14 = v14.Add(d)
		v15 = v15.Add(d)
		v16 = v16.Add(d)
		v17 = v17.Add(d)
		v18 = v18.Add(d)
		v19 = v19.Add(d)
		v20 = v20.Add(d)
		v21 = v21.Add(d)
		v22 = v22.Add(d)
		v23 = v23.Add(d)
		v24 = v24.Add(d)
		v25 = v25.Add(d)
		v26 = v26.Add(d)
		v27 = v27.Add(d)
		v28 = v28.Add(d)
		v29 = v29.Add(d)
		if iters&1023 == 0 {
			progress.Store(uint64(iters))
		}
	}

	var got [30][8]int64
	v0.StoreArray(&got[0])
	v1.StoreArray(&got[1])
	v2.StoreArray(&got[2])
	v3.StoreArray(&got[3])
	v4.StoreArray(&got[4])
	v5.StoreArray(&got[5])
	v6.StoreArray(&got[6])
	v7.StoreArray(&got[7])
	v8.StoreArray(&got[8])
	v9.StoreArray(&got[9])
	v10.StoreArray(&got[10])
	v11.StoreArray(&got[11])
	v12.StoreArray(&got[12])
	v13.StoreArray(&got[13])
	v14.StoreArray(&got[14])
	v15.StoreArray(&got[15])
	v16.StoreArray(&got[16])
	v17.StoreArray(&got[17])
	v18.StoreArray(&got[18])
	v19.StoreArray(&got[19])
	v20.StoreArray(&got[20])
	v21.StoreArray(&got[21])
	v22.StoreArray(&got[22])
	v23.StoreArray(&got[23])
	v24.StoreArray(&got[24])
	v25.StoreArray(&got[25])
	v26.StoreArray(&got[26])
	v27.StoreArray(&got[27])
	v28.StoreArray(&got[28])
	v29.StoreArray(&got[29])

	for i := range got {
		for j := range 8 {
			want := inits[i][j] + iters*deltaArr[j]
			if got[i][j] != want {
				return fmt.Errorf("Z reg %d lane %d after %d iters: got %#x, want %#x", i, j, iters, got[i][j], want)
			}
		}
	}
	return nil
}

// runX28MaskedLoop uses EVEX 128-bit masked adds to keep 29 Int64x2 values live
// across X0-X14 and X16-X31 (Hi16_ZMM) along with an opmask register while
// YMM_Hi128 and ZMM_Hi256 remain in INIT state.
//
//go:noinline
func runX28MaskedLoop(stop *atomic.Uint32, progress *atomic.Uint64, seed uint64, maskBits uint8) error {
	if !archsimd.X86.AVX512() {
		return nil
	}
	inits := makeInit2(seed, 28)
	deltaArr := [2]int64{1, 3}
	d := archsimd.LoadInt64x2Array(&deltaArr)
	m := archsimd.Mask64x2FromBits(maskBits)

	v0 := archsimd.LoadInt64x2Array(&inits[0])
	v1 := archsimd.LoadInt64x2Array(&inits[1])
	v2 := archsimd.LoadInt64x2Array(&inits[2])
	v3 := archsimd.LoadInt64x2Array(&inits[3])
	v4 := archsimd.LoadInt64x2Array(&inits[4])
	v5 := archsimd.LoadInt64x2Array(&inits[5])
	v6 := archsimd.LoadInt64x2Array(&inits[6])
	v7 := archsimd.LoadInt64x2Array(&inits[7])
	v8 := archsimd.LoadInt64x2Array(&inits[8])
	v9 := archsimd.LoadInt64x2Array(&inits[9])
	v10 := archsimd.LoadInt64x2Array(&inits[10])
	v11 := archsimd.LoadInt64x2Array(&inits[11])
	v12 := archsimd.LoadInt64x2Array(&inits[12])
	v13 := archsimd.LoadInt64x2Array(&inits[13])
	v14 := archsimd.LoadInt64x2Array(&inits[14])
	v15 := archsimd.LoadInt64x2Array(&inits[15])
	v16 := archsimd.LoadInt64x2Array(&inits[16])
	v17 := archsimd.LoadInt64x2Array(&inits[17])
	v18 := archsimd.LoadInt64x2Array(&inits[18])
	v19 := archsimd.LoadInt64x2Array(&inits[19])
	v20 := archsimd.LoadInt64x2Array(&inits[20])
	v21 := archsimd.LoadInt64x2Array(&inits[21])
	v22 := archsimd.LoadInt64x2Array(&inits[22])
	v23 := archsimd.LoadInt64x2Array(&inits[23])
	v24 := archsimd.LoadInt64x2Array(&inits[24])
	v25 := archsimd.LoadInt64x2Array(&inits[25])
	v26 := archsimd.LoadInt64x2Array(&inits[26])
	v27 := archsimd.LoadInt64x2Array(&inits[27])

	var iters int64
	for stop.Load() == 0 {
		iters++
		v0 = v0.Add(d).Merge(v0, m)
		v1 = v1.Add(d).Merge(v1, m)
		v2 = v2.Add(d).Merge(v2, m)
		v3 = v3.Add(d).Merge(v3, m)
		v4 = v4.Add(d).Merge(v4, m)
		v5 = v5.Add(d).Merge(v5, m)
		v6 = v6.Add(d).Merge(v6, m)
		v7 = v7.Add(d).Merge(v7, m)
		v8 = v8.Add(d).Merge(v8, m)
		v9 = v9.Add(d).Merge(v9, m)
		v10 = v10.Add(d).Merge(v10, m)
		v11 = v11.Add(d).Merge(v11, m)
		v12 = v12.Add(d).Merge(v12, m)
		v13 = v13.Add(d).Merge(v13, m)
		v14 = v14.Add(d).Merge(v14, m)
		v15 = v15.Add(d).Merge(v15, m)
		v16 = v16.Add(d).Merge(v16, m)
		v17 = v17.Add(d).Merge(v17, m)
		v18 = v18.Add(d).Merge(v18, m)
		v19 = v19.Add(d).Merge(v19, m)
		v20 = v20.Add(d).Merge(v20, m)
		v21 = v21.Add(d).Merge(v21, m)
		v22 = v22.Add(d).Merge(v22, m)
		v23 = v23.Add(d).Merge(v23, m)
		v24 = v24.Add(d).Merge(v24, m)
		v25 = v25.Add(d).Merge(v25, m)
		v26 = v26.Add(d).Merge(v26, m)
		v27 = v27.Add(d).Merge(v27, m)
		if iters&1023 == 0 {
			progress.Store(uint64(iters))
		}
	}

	var got [28][2]int64
	v0.StoreArray(&got[0])
	v1.StoreArray(&got[1])
	v2.StoreArray(&got[2])
	v3.StoreArray(&got[3])
	v4.StoreArray(&got[4])
	v5.StoreArray(&got[5])
	v6.StoreArray(&got[6])
	v7.StoreArray(&got[7])
	v8.StoreArray(&got[8])
	v9.StoreArray(&got[9])
	v10.StoreArray(&got[10])
	v11.StoreArray(&got[11])
	v12.StoreArray(&got[12])
	v13.StoreArray(&got[13])
	v14.StoreArray(&got[14])
	v15.StoreArray(&got[15])
	v16.StoreArray(&got[16])
	v17.StoreArray(&got[17])
	v18.StoreArray(&got[18])
	v19.StoreArray(&got[19])
	v20.StoreArray(&got[20])
	v21.StoreArray(&got[21])
	v22.StoreArray(&got[22])
	v23.StoreArray(&got[23])
	v24.StoreArray(&got[24])
	v25.StoreArray(&got[25])
	v26.StoreArray(&got[26])
	v27.StoreArray(&got[27])

	for i := range got {
		for j := range 2 {
			want := inits[i][j]
			if maskBits&(1<<j) != 0 {
				want += iters * deltaArr[j]
			}
			if got[i][j] != want {
				return fmt.Errorf("X28 reg %d lane %d after %d iters: got %#x, want %#x", i, j, iters, got[i][j], want)
			}
		}
	}
	return nil
}

// runMaskLoop keeps all 7 allocatable 64-bit opmask registers (K1-K7) live
// across masked 512-bit byte vector adds, checking every bit of every mask.
//
//go:noinline
func runMaskLoop(stop *atomic.Uint32, progress *atomic.Uint64, seed uint64) error {
	if !archsimd.X86.AVX512() {
		return nil
	}
	mBits := [7]uint64{
		seed ^ 0xaaaa_aaaa_aaaa_aaaa,
		seed ^ 0x5555_5555_5555_5555,
		seed ^ 0xcccc_cccc_cccc_cccc,
		seed ^ 0x3333_3333_3333_3333,
		seed ^ 0xf0f0_f0f0_f0f0_f0f0,
		seed ^ 0x0f0f_0f0f_0f0f_0f0f,
		seed ^ 0x9696_6969_a5a5_5a5a,
	}
	m0 := archsimd.Mask8x64FromBits(mBits[0])
	m1 := archsimd.Mask8x64FromBits(mBits[1])
	m2 := archsimd.Mask8x64FromBits(mBits[2])
	m3 := archsimd.Mask8x64FromBits(mBits[3])
	m4 := archsimd.Mask8x64FromBits(mBits[4])
	m5 := archsimd.Mask8x64FromBits(mBits[5])
	m6 := archsimd.Mask8x64FromBits(mBits[6])

	var inits [7][64]int8
	for i := range inits {
		for j := range 64 {
			inits[i][j] = int8(seed + uint64(i*64+j)*7)
		}
	}
	var deltaArr [64]int8
	for j := range 64 {
		deltaArr[j] = int8(2*j + 1) // odd so every lane changes every iteration
	}
	d := archsimd.LoadInt8x64Array(&deltaArr)

	v0 := archsimd.LoadInt8x64Array(&inits[0])
	v1 := archsimd.LoadInt8x64Array(&inits[1])
	v2 := archsimd.LoadInt8x64Array(&inits[2])
	v3 := archsimd.LoadInt8x64Array(&inits[3])
	v4 := archsimd.LoadInt8x64Array(&inits[4])
	v5 := archsimd.LoadInt8x64Array(&inits[5])
	v6 := archsimd.LoadInt8x64Array(&inits[6])

	var iters uint64
	for stop.Load() == 0 {
		iters++
		v0 = v0.Add(d).Merge(v0, m0)
		v1 = v1.Add(d).Merge(v1, m1)
		v2 = v2.Add(d).Merge(v2, m2)
		v3 = v3.Add(d).Merge(v3, m3)
		v4 = v4.Add(d).Merge(v4, m4)
		v5 = v5.Add(d).Merge(v5, m5)
		v6 = v6.Add(d).Merge(v6, m6)
		if iters&1023 == 0 {
			progress.Store(iters)
		}
	}

	var got [7][64]int8
	v0.StoreArray(&got[0])
	v1.StoreArray(&got[1])
	v2.StoreArray(&got[2])
	v3.StoreArray(&got[3])
	v4.StoreArray(&got[4])
	v5.StoreArray(&got[5])
	v6.StoreArray(&got[6])

	for i := range got {
		for j := range 64 {
			want := inits[i][j]
			if mBits[i]&(1<<uint(j)) != 0 {
				want += int8(iters) * deltaArr[j]
			}
			if got[i][j] != want {
				return fmt.Errorf("Mask reg %d lane %d after %d iters: got %#x, want %#x", i, j, iters, uint8(got[i][j]), uint8(want))
			}
		}
	}
	gotMasks := [7]uint64{
		m0.ToBits(),
		m1.ToBits(),
		m2.ToBits(),
		m3.ToBits(),
		m4.ToBits(),
		m5.ToBits(),
		m6.ToBits(),
	}
	for i := range gotMasks {
		if gotMasks[i] != mBits[i] {
			return fmt.Errorf("K reg %d after %d iters: got %#016x, want %#016x", i, iters, gotMasks[i], mBits[i])
		}
	}
	return nil
}

// runZAndMaskLoop keeps Z0-Z14, Z16-Z31, and K1-K7 live simultaneously.
//
//go:noinline
func runZAndMaskLoop(stop *atomic.Uint32, progress *atomic.Uint64, seed uint64, maskSeed uint64) error {
	if !archsimd.X86.AVX512() {
		return nil
	}
	inits := makeInit8(seed, 28)
	deltaArr := [8]int64{1, 3, 5, 7, 11, 13, 17, 19}
	d := archsimd.LoadInt64x8Array(&deltaArr)

	mBits := [7]uint8{
		uint8(maskSeed>>0) ^ 0xaa,
		uint8(maskSeed>>8) ^ 0x55,
		uint8(maskSeed>>16) ^ 0xcc,
		uint8(maskSeed>>24) ^ 0x33,
		uint8(maskSeed>>32) ^ 0xf0,
		uint8(maskSeed>>40) ^ 0x0f,
		uint8(maskSeed>>48) ^ 0x96,
	}
	m0 := archsimd.Mask64x8FromBits(mBits[0])
	m1 := archsimd.Mask64x8FromBits(mBits[1])
	m2 := archsimd.Mask64x8FromBits(mBits[2])
	m3 := archsimd.Mask64x8FromBits(mBits[3])
	m4 := archsimd.Mask64x8FromBits(mBits[4])
	m5 := archsimd.Mask64x8FromBits(mBits[5])
	m6 := archsimd.Mask64x8FromBits(mBits[6])

	v0 := archsimd.LoadInt64x8Array(&inits[0])
	v1 := archsimd.LoadInt64x8Array(&inits[1])
	v2 := archsimd.LoadInt64x8Array(&inits[2])
	v3 := archsimd.LoadInt64x8Array(&inits[3])
	v4 := archsimd.LoadInt64x8Array(&inits[4])
	v5 := archsimd.LoadInt64x8Array(&inits[5])
	v6 := archsimd.LoadInt64x8Array(&inits[6])
	v7 := archsimd.LoadInt64x8Array(&inits[7])
	v8 := archsimd.LoadInt64x8Array(&inits[8])
	v9 := archsimd.LoadInt64x8Array(&inits[9])
	v10 := archsimd.LoadInt64x8Array(&inits[10])
	v11 := archsimd.LoadInt64x8Array(&inits[11])
	v12 := archsimd.LoadInt64x8Array(&inits[12])
	v13 := archsimd.LoadInt64x8Array(&inits[13])
	v14 := archsimd.LoadInt64x8Array(&inits[14])
	v15 := archsimd.LoadInt64x8Array(&inits[15])
	v16 := archsimd.LoadInt64x8Array(&inits[16])
	v17 := archsimd.LoadInt64x8Array(&inits[17])
	v18 := archsimd.LoadInt64x8Array(&inits[18])
	v19 := archsimd.LoadInt64x8Array(&inits[19])
	v20 := archsimd.LoadInt64x8Array(&inits[20])
	v21 := archsimd.LoadInt64x8Array(&inits[21])
	v22 := archsimd.LoadInt64x8Array(&inits[22])
	v23 := archsimd.LoadInt64x8Array(&inits[23])
	v24 := archsimd.LoadInt64x8Array(&inits[24])
	v25 := archsimd.LoadInt64x8Array(&inits[25])
	v26 := archsimd.LoadInt64x8Array(&inits[26])
	v27 := archsimd.LoadInt64x8Array(&inits[27])

	var iters int64
	for stop.Load() == 0 {
		iters++
		v0 = v0.Add(d).Merge(v0, m0)
		v1 = v1.Add(d).Merge(v1, m1)
		v2 = v2.Add(d).Merge(v2, m2)
		v3 = v3.Add(d).Merge(v3, m3)
		v4 = v4.Add(d).Merge(v4, m4)
		v5 = v5.Add(d).Merge(v5, m5)
		v6 = v6.Add(d).Merge(v6, m6)
		v7 = v7.Add(d).Merge(v7, m0)
		v8 = v8.Add(d).Merge(v8, m1)
		v9 = v9.Add(d).Merge(v9, m2)
		v10 = v10.Add(d).Merge(v10, m3)
		v11 = v11.Add(d).Merge(v11, m4)
		v12 = v12.Add(d).Merge(v12, m5)
		v13 = v13.Add(d).Merge(v13, m6)
		v14 = v14.Add(d).Merge(v14, m0)
		v15 = v15.Add(d).Merge(v15, m1)
		v16 = v16.Add(d).Merge(v16, m2)
		v17 = v17.Add(d).Merge(v17, m3)
		v18 = v18.Add(d).Merge(v18, m4)
		v19 = v19.Add(d).Merge(v19, m5)
		v20 = v20.Add(d).Merge(v20, m6)
		v21 = v21.Add(d).Merge(v21, m0)
		v22 = v22.Add(d).Merge(v22, m1)
		v23 = v23.Add(d).Merge(v23, m2)
		v24 = v24.Add(d).Merge(v24, m3)
		v25 = v25.Add(d).Merge(v25, m4)
		v26 = v26.Add(d).Merge(v26, m5)
		v27 = v27.Add(d).Merge(v27, m6)
		if iters&1023 == 0 {
			progress.Store(uint64(iters))
		}
	}

	var got [28][8]int64
	v0.StoreArray(&got[0])
	v1.StoreArray(&got[1])
	v2.StoreArray(&got[2])
	v3.StoreArray(&got[3])
	v4.StoreArray(&got[4])
	v5.StoreArray(&got[5])
	v6.StoreArray(&got[6])
	v7.StoreArray(&got[7])
	v8.StoreArray(&got[8])
	v9.StoreArray(&got[9])
	v10.StoreArray(&got[10])
	v11.StoreArray(&got[11])
	v12.StoreArray(&got[12])
	v13.StoreArray(&got[13])
	v14.StoreArray(&got[14])
	v15.StoreArray(&got[15])
	v16.StoreArray(&got[16])
	v17.StoreArray(&got[17])
	v18.StoreArray(&got[18])
	v19.StoreArray(&got[19])
	v20.StoreArray(&got[20])
	v21.StoreArray(&got[21])
	v22.StoreArray(&got[22])
	v23.StoreArray(&got[23])
	v24.StoreArray(&got[24])
	v25.StoreArray(&got[25])
	v26.StoreArray(&got[26])
	v27.StoreArray(&got[27])

	for i := range got {
		mb := mBits[i%len(mBits)]
		for j := range 8 {
			want := inits[i][j]
			if mb&(1<<uint(j)) != 0 {
				want += iters * deltaArr[j]
			}
			if got[i][j] != want {
				return fmt.Errorf("Z+Mask Z reg %d lane %d after %d iters: got %#x, want %#x", i, j, iters, got[i][j], want)
			}
		}
	}

	mGot := [7]uint8{
		m0.ToBits(),
		m1.ToBits(),
		m2.ToBits(),
		m3.ToBits(),
		m4.ToBits(),
		m5.ToBits(),
		m6.ToBits(),
	}
	for i := range mGot {
		if mGot[i] != mBits[i] {
			return fmt.Errorf("Z+Mask K reg %d after %d iters: got %#02x, want %#02x", i, iters, mGot[i], mBits[i])
		}
	}
	return nil
}
