// +build !android

package runtime

import "unsafe"

func writeErr(b []byte) {
	write(2, unsafe.Pointer(&b[0]), int32(len(b)))
}
