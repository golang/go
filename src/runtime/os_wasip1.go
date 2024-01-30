// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package runtime

import "unsafe"

// GOARCH=wasm currently has 64 bits pointers, but the WebAssembly host expects
// pointers to be 32 bits so we use this type alias to represent pointers in
// structs and arrays passed as arguments to WASI functions.
//
// Note that the use of an integer type prevents the compiler from tracking
// pointers passed to WASI functions, so we must use KeepAlive to explicitly
// retain the objects that could otherwise be reclaimed by the GC.
type uintptr32 = uint32

// https://github.com/WebAssembly/WASI/blob/a2b96e81c0586125cc4dc79a5be0b78d9a059925/legacy/preview1/docs.md#-size-u32
type size = uint32

// https://github.com/WebAssembly/WASI/blob/a2b96e81c0586125cc4dc79a5be0b78d9a059925/legacy/preview1/docs.md#-errno-variant
type errno = uint32

// https://github.com/WebAssembly/WASI/blob/a2b96e81c0586125cc4dc79a5be0b78d9a059925/legacy/preview1/docs.md#-filesize-u64
type filesize = uint64

// https://github.com/WebAssembly/WASI/blob/a2b96e81c0586125cc4dc79a5be0b78d9a059925/legacy/preview1/docs.md#-timestamp-u64
type timestamp = uint64

// https://github.com/WebAssembly/WASI/blob/a2b96e81c0586125cc4dc79a5be0b78d9a059925/legacy/preview1/docs.md#-clockid-variant
type clockid = uint32

const (
	clockRealtime  clockid = 0
	clockMonotonic clockid = 1
)

// https://github.com/WebAssembly/WASI/blob/a2b96e81c0586125cc4dc79a5be0b78d9a059925/legacy/preview1/docs.md#-iovec-record
type iovec struct {
	buf    uintptr32
	bufLen size
}

//go:wasmimport wasi_snapshot_preview1 proc_exit
func exit(code int32)

//go:wasmimport wasi_snapshot_preview1 args_get
//go:noescape
func args_get(argv, argvBuf unsafe.Pointer) errno

//go:wasmimport wasi_snapshot_preview1 args_sizes_get
//go:noescape
func args_sizes_get(argc, argvBufLen unsafe.Pointer) errno

//go:wasmimport wasi_snapshot_preview1 clock_time_get
//go:noescape
func clock_time_get(clock_id clockid, precision timestamp, time unsafe.Pointer) errno

//go:wasmimport wasi_snapshot_preview1 environ_get
//go:noescape
func environ_get(environ, environBuf unsafe.Pointer) errno

//go:wasmimport wasi_snapshot_preview1 environ_sizes_get
//go:noescape
func environ_sizes_get(environCount, environBufLen unsafe.Pointer) errno

//go:wasmimport wasi_snapshot_preview1 fd_write
//go:noescape
func fd_write(fd int32, iovs unsafe.Pointer, iovsLen size, nwritten unsafe.Pointer) errno

//go:wasmimport wasi_snapshot_preview1 random_get
//go:noescape
func random_get(buf unsafe.Pointer, bufLen size) errno

type eventtype = uint8

const (
	eventtypeClock eventtype = iota
	eventtypeFdRead
	eventtypeFdWrite
)

type eventrwflags = uint16

const (
	fdReadwriteHangup eventrwflags = 1 << iota
)

type userdata = uint64

// The go:wasmimport directive currently does not accept values of type uint16
// in arguments or returns of the function signature. Most WASI imports return
// an errno value, which we have to define as uint32 because of that limitation.
// However, the WASI errno type is intended to be a 16 bits integer, and in the
// event struct the error field should be of type errno. If we used the errno
// type for the error field it would result in a mismatching field alignment and
// struct size because errno is declared as a 32 bits type, so we declare the
// error field as a plain uint16.
type event struct {
	userdata    userdata
	error       uint16
	typ         eventtype
	fdReadwrite eventFdReadwrite
}

type eventFdReadwrite struct {
	nbytes filesize
	flags  eventrwflags
}

type subclockflags = uint16

const (
	subscriptionClockAbstime subclockflags = 1 << iota
)

type subscriptionClock struct {
	id        clockid
	timeout   timestamp
	precision timestamp
	flags     subclockflags
}

type subscriptionFdReadwrite struct {
	fd int32
}

type subscription struct {
	userdata userdata
	u        subscriptionUnion
}

type subscriptionUnion [5]uint64

func (u *subscriptionUnion) eventtype() *eventtype {
	return (*eventtype)(unsafe.Pointer(&u[0]))
}

func (u *subscriptionUnion) subscriptionClock() *subscriptionClock {
	return (*subscriptionClock)(unsafe.Pointer(&u[1]))
}

func (u *subscriptionUnion) subscriptionFdReadwrite() *subscriptionFdReadwrite {
	return (*subscriptionFdReadwrite)(unsafe.Pointer(&u[1]))
}

//go:wasmimport wasi_snapshot_preview1 poll_oneoff
//go:noescape
func poll_oneoff(in, out unsafe.Pointer, nsubscriptions size, nevents unsafe.Pointer) errno

func write1(fd uintptr, p unsafe.Pointer, n int32) int32 {
	iov := iovec{
		buf:    uintptr32(uintptr(p)),
		bufLen: size(n),
	}
	var nwritten size
	if fd_write(int32(fd), unsafe.Pointer(&iov), 1, unsafe.Pointer(&nwritten)) != 0 {
		throw("fd_write failed")
	}
	return int32(nwritten)
}

func usleep(usec uint32) {
	var in subscription
	var out event
	var nevents size

	eventtype := in.u.eventtype()
	*eventtype = eventtypeClock

	subscription := in.u.subscriptionClock()
	subscription.id = clockMonotonic
	subscription.timeout = timestamp(usec) * 1e3
	subscription.precision = 1e3

	if poll_oneoff(unsafe.Pointer(&in), unsafe.Pointer(&out), 1, unsafe.Pointer(&nevents)) != 0 {
		throw("wasi_snapshot_preview1.poll_oneoff")
	}
}

func readRandom(r []byte) int {
	if random_get(unsafe.Pointer(&r[0]), size(len(r))) != 0 {
		return 0
	}
	return len(r)
}

func goenvs() {
	// arguments
	var argc size
	var argvBufLen size
	if args_sizes_get(unsafe.Pointer(&argc), unsafe.Pointer(&argvBufLen)) != 0 {
		throw("args_sizes_get failed")
	}

	argslice = make([]string, argc)
	if argc > 0 {
		argv := make([]uintptr32, argc)
		argvBuf := make([]byte, argvBufLen)
		if args_get(unsafe.Pointer(&argv[0]), unsafe.Pointer(&argvBuf[0])) != 0 {
			throw("args_get failed")
		}

		for i := range argslice {
			start := argv[i] - uintptr32(uintptr(unsafe.Pointer(&argvBuf[0])))
			end := start
			for argvBuf[end] != 0 {
				end++
			}
			argslice[i] = string(argvBuf[start:end])
		}
	}

	// environment
	var environCount size
	var environBufLen size
	if environ_sizes_get(unsafe.Pointer(&environCount), unsafe.Pointer(&environBufLen)) != 0 {
		throw("environ_sizes_get failed")
	}

	envs = make([]string, environCount)
	if environCount > 0 {
		environ := make([]uintptr32, environCount)
		environBuf := make([]byte, environBufLen)
		if environ_get(unsafe.Pointer(&environ[0]), unsafe.Pointer(&environBuf[0])) != 0 {
			throw("environ_get failed")
		}

		for i := range envs {
			start := environ[i] - uintptr32(uintptr(unsafe.Pointer(&environBuf[0])))
			end := start
			for environBuf[end] != 0 {
				end++
			}
			envs[i] = string(environBuf[start:end])
		}
	}
}

func walltime() (sec int64, nsec int32) {
	return walltime1()
}

func walltime1() (sec int64, nsec int32) {
	var time timestamp
	if clock_time_get(clockRealtime, 0, unsafe.Pointer(&time)) != 0 {
		throw("clock_time_get failed")
	}
	return int64(time / 1000000000), int32(time % 1000000000)
}

func nanotime1() int64 {
	var time timestamp
	if clock_time_get(clockMonotonic, 0, unsafe.Pointer(&time)) != 0 {
		throw("clock_time_get failed")
	}
	return int64(time)
}
