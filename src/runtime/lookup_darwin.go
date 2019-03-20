package runtime

import (
	"unsafe"
)

//go:linkname res_ninit net.res_ninit
//go:nosplit
//go:cgo_unsafe_args
func res_ninit(statp *[71]uint64) int32 {
	return libcCall(unsafe.Pointer(funcPC(res_ninit_trampoline)), unsafe.Pointer(&statp))
}
func res_ninit_trampoline()

//go:linkname res_nsearch net.res_nsearch
//go:nosplit
//go:cgo_unsafe_args
func res_nsearch(statp *[71]uint64, dname *byte, class int32, rtype int32, answer *byte, anslen int32) int32 {
	return libcCall(unsafe.Pointer(funcPC(res_nsearch_trampoline)), unsafe.Pointer(&statp))
}
func res_nsearch_trampoline()

//go:cgo_import_dynamic libc_res_nsearch res_nsearch "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_res_ninit res_ninit "/usr/lib/libSystem.B.dylib"
