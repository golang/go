package runtime

import (
	"unsafe"
)

//go:linkname res_search net.res_search
//go:nosplit
//go:cgo_unsafe_args
func res_search(name *byte, class int32, rtype int32, answer *byte, anslen int32) int32 {
	return libcCall(unsafe.Pointer(funcPC(res_search_trampoline)), unsafe.Pointer(&name))
}
func res_search_trampoline()

//go:cgo_import_dynamic libc_res_search res_search "/usr/lib/libSystem.B.dylib"
