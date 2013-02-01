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
