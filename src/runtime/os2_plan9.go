// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Plan 9-specific system calls

package runtime

// open
const (
	_OREAD   = 0
	_OWRITE  = 1
	_ORDWR   = 2
	_OEXEC   = 3
	_OTRUNC  = 16
	_OCEXEC  = 32
	_ORCLOSE = 64
	_OEXCL   = 0x1000
)

// rfork
const (
	_RFNAMEG  = 1 << 0
	_RFENVG   = 1 << 1
	_RFFDG    = 1 << 2
	_RFNOTEG  = 1 << 3
	_RFPROC   = 1 << 4
	_RFMEM    = 1 << 5
	_RFNOWAIT = 1 << 6
	_RFCNAMEG = 1 << 10
	_RFCENVG  = 1 << 11
	_RFCFDG   = 1 << 12
	_RFREND   = 1 << 13
	_RFNOMNT  = 1 << 14
)

// notify
const (
	_NCONT = 0
	_NDFLT = 1
)

type uinptr _Plink

type tos struct {
	prof struct { // Per process profiling
		pp    *_Plink // known to be 0(ptr)
		next  *_Plink // known to be 4(ptr)
		last  *_Plink
		first *_Plink
		pid   uint32
		what  uint32
	}
	cyclefreq uint64 // cycle clock frequency if there is one, 0 otherwise
	kcycles   int64  // cycles spent in kernel
	pcycles   int64  // cycles spent in process (kernel + user)
	pid       uint32 // might as well put the pid here
	clock     uint32
	// top of stack is here
}

const (
	_NSIG   = 14  // number of signals in sigtable array
	_ERRMAX = 128 // max length of note string

	// Notes in runtime·sigtab that are handled by runtime·sigpanic.
	_SIGRFAULT = 2
	_SIGWFAULT = 3
	_SIGINTDIV = 4
	_SIGFLOAT  = 5
	_SIGTRAP   = 6
	_SIGPROF   = 0 // dummy value defined for badsignal
	_SIGQUIT   = 0 // dummy value defined for sighandler
)
