package runtime

import (
	"unsafe"
)

//go:linkname res_init net.res_init
//go:nosplit
//go:cgo_unsafe_args
func res_init(statp *[71]uint64) int32 {
	return libcCall(unsafe.Pointer(funcPC(res_init_trampoline)), unsafe.Pointer(&statp))
}
func res_init_trampoline()

//go:linkname res_nsearch net.res_nsearch
//go:nosplit
//go:cgo_unsafe_args
func res_nsearch(statp *[71]uint64, dname *byte, class int32, rtype int32, answer *byte, anslen int32) int32 {
	return libcCall(unsafe.Pointer(funcPC(res_nsearch_trampoline)), unsafe.Pointer(&statp))
}
func res_nsearch_trampoline()

//go:cgo_import_dynamic libc_res_nsearch res_nsearch "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_res_init res_init "/usr/lib/libSystem.B.dylib"
