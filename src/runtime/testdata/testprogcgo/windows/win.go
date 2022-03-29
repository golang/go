package windows

/*
#cgo amd64 386 CFLAGS: -mnop-fun-dllimport

#include <windows.h>

DWORD agetthread() {
	return GetCurrentThreadId();
}
*/
import "C"

func GetThread() uint32 {
	return uint32(C.agetthread())
}
