package runtime

import "unsafe"

var (
	writeHeader = []byte{6 /* ANDROID_LOG_ERROR */, 'G', 'o', 0}
	writePath   = []byte("/dev/log/main\x00")
	writeFD     uintptr
	writeBuf    [1024]byte
	writePos    int
)

func writeErr(b []byte) {
	// Log format: "<priority 1 byte><tag n bytes>\x00<message m bytes>\x00"
	// The entire log needs to be delivered in a single syscall (the NDK
	// does this with writev). Each log is its own line, so we need to
	// buffer writes until we see a newline.
	if writeFD == 0 {
		writeFD = uintptr(open(&writePath[0], 0x1 /* O_WRONLY */, 0))
		if writeFD == 0 {
			// It is hard to do anything here. Write to stderr just
			// in case user has root on device and has run
			//	adb shell setprop log.redirect-stdio true
			msg := []byte("runtime: cannot open /dev/log/main\x00")
			write(2, unsafe.Pointer(&msg[0]), int32(len(msg)))
			exit(2)
		}
		copy(writeBuf[:], writeHeader)
	}
	dst := writeBuf[len(writeHeader):]
	for _, v := range b {
		if v == 0 { // android logging won't print a zero byte
			v = '0'
		}
		dst[writePos] = v
		writePos++
		if v == '\n' || writePos == len(dst)-1 {
			dst[writePos] = 0
			write(writeFD, unsafe.Pointer(&writeBuf[0]), int32(len(writeHeader)+writePos))
			memclrBytes(dst)
			writePos = 0
		}
	}
}
