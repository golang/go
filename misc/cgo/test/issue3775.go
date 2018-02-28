// +build !android

package cgotest

/*
void lockOSThreadCallback(void);
inline static void lockOSThreadC(void)
{
        lockOSThreadCallback();
}
int usleep(unsigned usec);
*/
import "C"

import (
	"runtime"
	"testing"
)

func init() {
	// Same as test3775 but run during init so that
	// there are two levels of internal runtime lock
	// (1 for init, 1 for cgo).
	// This would have been broken by CL 11663043.
	C.lockOSThreadC()
}

func test3775(t *testing.T) {
	// Used to panic because of the UnlockOSThread below.
	C.lockOSThreadC()
}

//export lockOSThreadCallback
func lockOSThreadCallback() {
	runtime.LockOSThread()
	runtime.UnlockOSThread()
	go C.usleep(10000)
	runtime.Gosched()
}
