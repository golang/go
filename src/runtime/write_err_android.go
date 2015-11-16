// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

var (
	writeHeader = []byte{6 /* ANDROID_LOG_ERROR */, 'G', 'o', 0}
	writePath   = []byte("/dev/log/main\x00")
	writeLogd   = []byte("/dev/socket/logdw\x00")

	// guarded by printlock/printunlock.
	writeFD  uintptr
	writeBuf [1024]byte
	writePos int
)

// Prior to Android-L, logging was done through writes to /dev/log files implemented
// in kernel ring buffers. In Android-L, those /dev/log files are no longer
// accessible and logging is done through a centralized user-mode logger, logd.
//
// https://android.googlesource.com/platform/system/core/+/master/liblog/logd_write.c
type loggerType int32

const (
	unknown loggerType = iota
	legacy
	logd
	// TODO(hakim): logging for emulator?
)

var logger loggerType

func writeErr(b []byte) {
	if logger == unknown {
		// Use logd if /dev/socket/logdw is available.
		if v := uintptr(access(&writeLogd[0], 0x02 /* W_OK */)); v == 0 {
			logger = logd
			initLogd()
		} else {
			logger = legacy
			initLegacy()
		}
	}

	// Write to stderr for command-line programs.
	write(2, unsafe.Pointer(&b[0]), int32(len(b)))

	// Log format: "<header>\x00<message m bytes>\x00"
	//
	// <header>
	//   In legacy mode: "<priority 1 byte><tag n bytes>".
	//   In logd mode: "<android_log_header_t 11 bytes><priority 1 byte><tag n bytes>"
	//
	// The entire log needs to be delivered in a single syscall (the NDK
	// does this with writev). Each log is its own line, so we need to
	// buffer writes until we see a newline.
	var hlen int
	switch logger {
	case logd:
		hlen = writeLogdHeader()
	case legacy:
		hlen = len(writeHeader)
	}

	dst := writeBuf[hlen:]
	for _, v := range b {
		if v == 0 { // android logging won't print a zero byte
			v = '0'
		}
		dst[writePos] = v
		writePos++
		if v == '\n' || writePos == len(dst)-1 {
			dst[writePos] = 0
			write(writeFD, unsafe.Pointer(&writeBuf[0]), int32(hlen+writePos))
			memclrBytes(dst)
			writePos = 0
		}
	}
}

func initLegacy() {
	// In legacy mode, logs are written to /dev/log/main
	writeFD = uintptr(open(&writePath[0], 0x1 /* O_WRONLY */, 0))
	if writeFD == 0 {
		// It is hard to do anything here. Write to stderr just
		// in case user has root on device and has run
		//	adb shell setprop log.redirect-stdio true
		msg := []byte("runtime: cannot open /dev/log/main\x00")
		write(2, unsafe.Pointer(&msg[0]), int32(len(msg)))
		exit(2)
	}

	// Prepopulate the invariant header part.
	copy(writeBuf[:len(writeHeader)], writeHeader)
}

// used in initLogdWrite but defined here to avoid heap allocation.
var logdAddr sockaddr_un

func initLogd() {
	// In logd mode, logs are sent to the logd via a unix domain socket.
	logdAddr.family = _AF_UNIX
	copy(logdAddr.path[:], writeLogd)

	// We are not using non-blocking I/O because writes taking this path
	// are most likely triggered by panic, we cannot think of the advantage of
	// non-blocking I/O for panic but see disadvantage (dropping panic message),
	// and blocking I/O simplifies the code a lot.
	fd := socket(_AF_UNIX, _SOCK_DGRAM|_O_CLOEXEC, 0)
	if fd < 0 {
		msg := []byte("runtime: cannot create a socket for logging\x00")
		write(2, unsafe.Pointer(&msg[0]), int32(len(msg)))
		exit(2)
	}

	errno := connect(fd, unsafe.Pointer(&logdAddr), int32(unsafe.Sizeof(logdAddr)))
	if errno < 0 {
		msg := []byte("runtime: cannot connect to /dev/socket/logdw\x00")
		write(2, unsafe.Pointer(&msg[0]), int32(len(msg)))
		// TODO(hakim): or should we just close fd and hope for better luck next time?
		exit(2)
	}
	writeFD = uintptr(fd)

	// Prepopulate invariant part of the header.
	// The first 11 bytes will be populated later in writeLogdHeader.
	copy(writeBuf[11:11+len(writeHeader)], writeHeader)
}

// writeLogdHeader populates the header and returns the length of the payload.
func writeLogdHeader() int {
	hdr := writeBuf[:11]

	// The first 11 bytes of the header corresponds to android_log_header_t
	// as defined in system/core/include/private/android_logger.h
	//   hdr[0] log type id (unsigned char), defined in <log/log.h>
	//   hdr[1:2] tid (uint16_t)
	//   hdr[3:11] log_time defined in <log/log_read.h>
	//      hdr[3:7] sec unsigned uint32, little endian.
	//      hdr[7:11] nsec unsigned uint32, little endian.
	hdr[0] = 0 // LOG_ID_MAIN
	sec, nsec := time_now()
	packUint32(hdr[3:7], uint32(sec))
	packUint32(hdr[7:11], uint32(nsec))

	// TODO(hakim):  hdr[1:2] = gettid?

	return 11 + len(writeHeader)
}

func packUint32(b []byte, v uint32) {
	// little-endian.
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
}
