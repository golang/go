// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"math/bits"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

type mutatorRand interface {
	uint32() uint32
	intn(int) int
	uint32n(uint32) uint32
	exp2() int
	bool() bool

	save(randState, randInc *uint64)
	restore(randState, randInc uint64)
}

// The functions in pcg implement a 32 bit PRNG with a 64 bit period: pcg xsh rr
// 64 32. See https://www.pcg-random.org/ for more information. This
// implementation is geared specifically towards the needs of fuzzing: Simple
// creation and use, no reproducibility, no concurrency safety, just the
// necessary methods, optimized for speed.

var globalInc uint64 // PCG stream

const multiplier uint64 = 6364136223846793005

// pcgRand is a PRNG. It should not be copied or shared. No Rand methods are
// concurrency safe.
type pcgRand struct {
	noCopy noCopy // help avoid mistakes: ask vet to ensure that we don't make a copy
	state  uint64
	inc    uint64
}

func godebugSeed() *int {
	debug := strings.Split(os.Getenv("GODEBUG"), ",")
	for _, f := range debug {
		if strings.HasPrefix(f, "fuzzseed=") {
			seed, err := strconv.Atoi(strings.TrimPrefix(f, "fuzzseed="))
			if err != nil {
				panic("malformed fuzzseed")
			}
			return &seed
		}
	}
	return nil
}

// newPcgRand generates a new, seeded Rand, ready for use.
func newPcgRand() *pcgRand {
	r := new(pcgRand)
	now := uint64(time.Now().UnixNano())
	if seed := godebugSeed(); seed != nil {
		now = uint64(*seed)
	}
	inc := atomic.AddUint64(&globalInc, 1)
	r.state = now
	r.inc = (inc << 1) | 1
	r.step()
	r.state += now
	r.step()
	return r
}

func (r *pcgRand) step() {
	r.state *= multiplier
	r.state += r.inc
}

func (r *pcgRand) save(randState, randInc *uint64) {
	*randState = r.state
	*randInc = r.inc
}

func (r *pcgRand) restore(randState, randInc uint64) {
	r.state = randState
	r.inc = randInc
}

// uint32 returns a pseudo-random uint32.
func (r *pcgRand) uint32() uint32 {
	x := r.state
	r.step()
	return bits.RotateLeft32(uint32(((x>>18)^x)>>27), -int(x>>59))
}

// intn returns a pseudo-random number in [0, n).
// n must fit in a uint32.
func (r *pcgRand) intn(n int) int {
	if int(uint32(n)) != n {
		panic("large Intn")
	}
	return int(r.uint32n(uint32(n)))
}

// uint32n returns a pseudo-random number in [0, n).
//
// For implementation details, see:
// https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction
// https://lemire.me/blog/2016/06/30/fast-random-shuffling
func (r *pcgRand) uint32n(n uint32) uint32 {
	v := r.uint32()
	prod := uint64(v) * uint64(n)
	low := uint32(prod)
	if low < n {
		thresh := uint32(-int32(n)) % n
		for low < thresh {
			v = r.uint32()
			prod = uint64(v) * uint64(n)
			low = uint32(prod)
		}
	}
	return uint32(prod >> 32)
}

// exp2 generates n with probability 1/2^(n+1).
func (r *pcgRand) exp2() int {
	return bits.TrailingZeros32(r.uint32())
}

// bool generates a random bool.
func (r *pcgRand) bool() bool {
	return r.uint32()&1 == 0
}

// noCopy may be embedded into structs which must not be copied
// after the first use.
//
// See https://golang.org/issues/8005#issuecomment-190753527
// for details.
type noCopy struct{}

// lock is a no-op used by -copylocks checker from `go vet`.
func (*noCopy) lock()   {}
func (*noCopy) unlock() {}
