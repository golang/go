// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Random number generation

package runtime

import (
	"internal/chacha8rand"
	"internal/goarch"
	"runtime/internal/math"
	"unsafe"
	_ "unsafe" // for go:linkname
)

// OS-specific startup can set startupRand if the OS passes
// random data to the process at startup time.
// For example Linux passes 16 bytes in the auxv vector.
var startupRand []byte

// globalRand holds the global random state.
// It is only used at startup and for creating new m's.
// Otherwise the per-m random state should be used
// by calling goodrand.
var globalRand struct {
	lock  mutex
	seed  [32]byte
	state chacha8rand.State
	init  bool
}

var readRandomFailed bool

// randinit initializes the global random state.
// It must be called before any use of grand.
func randinit() {
	lock(&globalRand.lock)
	if globalRand.init {
		fatal("randinit twice")
	}

	seed := &globalRand.seed
	if startupRand != nil {
		for i, c := range startupRand {
			seed[i%len(seed)] ^= c
		}
		clear(startupRand)
		startupRand = nil
	} else {
		if readRandom(seed[:]) != len(seed) {
			// readRandom should never fail, but if it does we'd rather
			// not make Go binaries completely unusable, so make up
			// some random data based on the current time.
			readRandomFailed = true
			readTimeRandom(seed[:])
		}
	}
	globalRand.state.Init(*seed)
	clear(seed[:])
	globalRand.init = true
	unlock(&globalRand.lock)
}

// readTimeRandom stretches any entropy in the current time
// into entropy the length of r and XORs it into r.
// This is a fallback for when readRandom does not read
// the full requested amount.
// Whatever entropy r already contained is preserved.
func readTimeRandom(r []byte) {
	// Inspired by wyrand.
	// An earlier version of this code used getg().m.procid as well,
	// but note that this is called so early in startup that procid
	// is not initialized yet.
	v := uint64(nanotime())
	for len(r) > 0 {
		v ^= 0xa0761d6478bd642f
		v *= 0xe7037ed1a0b428db
		size := 8
		if len(r) < 8 {
			size = len(r)
		}
		for i := 0; i < size; i++ {
			r[i] ^= byte(v >> (8 * i))
		}
		r = r[size:]
		v = v>>32 | v<<32
	}
}

// bootstrapRand returns a random uint64 from the global random generator.
func bootstrapRand() uint64 {
	lock(&globalRand.lock)
	if !globalRand.init {
		fatal("randinit missed")
	}
	for {
		if x, ok := globalRand.state.Next(); ok {
			unlock(&globalRand.lock)
			return x
		}
		globalRand.state.Refill()
	}
}

// bootstrapRandReseed reseeds the bootstrap random number generator,
// clearing from memory any trace of previously returned random numbers.
func bootstrapRandReseed() {
	lock(&globalRand.lock)
	if !globalRand.init {
		fatal("randinit missed")
	}
	globalRand.state.Reseed()
	unlock(&globalRand.lock)
}

// rand32 is uint32(rand()), called from compiler-generated code.
//
//go:nosplit
func rand32() uint32 {
	return uint32(rand())
}

// rand returns a random uint64 from the per-m chacha8 state.
// Do not change signature: used via linkname from other packages.
//
//go:nosplit
//go:linkname rand
func rand() uint64 {
	// Note: We avoid acquirem here so that in the fast path
	// there is just a getg, an inlined c.Next, and a return.
	// The performance difference on a 16-core AMD is
	// 3.7ns/call this way versus 4.3ns/call with acquirem (+16%).
	mp := getg().m
	c := &mp.chacha8
	for {
		// Note: c.Next is marked nosplit,
		// so we don't need to use mp.locks
		// on the fast path, which is that the
		// first attempt succeeds.
		x, ok := c.Next()
		if ok {
			return x
		}
		mp.locks++ // hold m even though c.Refill may do stack split checks
		c.Refill()
		mp.locks--
	}
}

// mrandinit initializes the random state of an m.
func mrandinit(mp *m) {
	var seed [4]uint64
	for i := range seed {
		seed[i] = bootstrapRand()
	}
	bootstrapRandReseed() // erase key we just extracted
	mp.chacha8.Init64(seed)
	mp.cheaprand = rand()
}

// randn is like rand() % n but faster.
// Do not change signature: used via linkname from other packages.
//
//go:nosplit
//go:linkname randn
func randn(n uint32) uint32 {
	// See https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
	return uint32((uint64(uint32(rand())) * uint64(n)) >> 32)
}

// cheaprand is a non-cryptographic-quality 32-bit random generator
// suitable for calling at very high frequency (such as during scheduling decisions)
// and at sensitive moments in the runtime (such as during stack unwinding).
// it is "cheap" in the sense of both expense and quality.
//
// cheaprand must not be exported to other packages:
// the rule is that other packages using runtime-provided
// randomness must always use rand.
//
// cheaprand should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/bytedance/gopkg
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname cheaprand
//go:nosplit
func cheaprand() uint32 {
	mp := getg().m
	// Implement wyrand: https://github.com/wangyi-fudan/wyhash
	// Only the platform that math.Mul64 can be lowered
	// by the compiler should be in this list.
	if goarch.IsAmd64|goarch.IsArm64|goarch.IsPpc64|
		goarch.IsPpc64le|goarch.IsMips64|goarch.IsMips64le|
		goarch.IsS390x|goarch.IsRiscv64|goarch.IsLoong64 == 1 {
		mp.cheaprand += 0xa0761d6478bd642f
		hi, lo := math.Mul64(mp.cheaprand, mp.cheaprand^0xe7037ed1a0b428db)
		return uint32(hi ^ lo)
	}

	// Implement xorshift64+: 2 32-bit xorshift sequences added together.
	// Shift triplet [17,7,16] was calculated as indicated in Marsaglia's
	// Xorshift paper: https://www.jstatsoft.org/article/view/v008i14/xorshift.pdf
	// This generator passes the SmallCrush suite, part of TestU01 framework:
	// http://simul.iro.umontreal.ca/testu01/tu01.html
	t := (*[2]uint32)(unsafe.Pointer(&mp.cheaprand))
	s1, s0 := t[0], t[1]
	s1 ^= s1 << 17
	s1 = s1 ^ s0 ^ s1>>7 ^ s0>>16
	t[0], t[1] = s0, s1
	return s0 + s1
}

// cheaprand64 is a non-cryptographic-quality 63-bit random generator
// suitable for calling at very high frequency (such as during sampling decisions).
// it is "cheap" in the sense of both expense and quality.
//
// cheaprand64 must not be exported to other packages:
// the rule is that other packages using runtime-provided
// randomness must always use rand.
//
// cheaprand64 should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/zhangyunhao116/fastrand
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname cheaprand64
//go:nosplit
func cheaprand64() int64 {
	return int64(cheaprand())<<31 ^ int64(cheaprand())
}

// cheaprandn is like cheaprand() % n but faster.
//
// cheaprandn must not be exported to other packages:
// the rule is that other packages using runtime-provided
// randomness must always use randn.
//
// cheaprandn should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/phuslu/log
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname cheaprandn
//go:nosplit
func cheaprandn(n uint32) uint32 {
	// See https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
	return uint32((uint64(cheaprand()) * uint64(n)) >> 32)
}

// Too much legacy code has go:linkname references
// to runtime.fastrand and friends, so keep these around for now.
// Code should migrate to math/rand/v2.Uint64,
// which is just as fast, but that's only available in Go 1.22+.
// It would be reasonable to remove these in Go 1.24.
// Do not call these from package runtime.

//go:linkname legacy_fastrand runtime.fastrand
func legacy_fastrand() uint32 {
	return uint32(rand())
}

//go:linkname legacy_fastrandn runtime.fastrandn
func legacy_fastrandn(n uint32) uint32 {
	return randn(n)
}

//go:linkname legacy_fastrand64 runtime.fastrand64
func legacy_fastrand64() uint64 {
	return rand()
}
