package runtime

import (
	"unsafe"
)

//go:linkname res_init net.res_init
//go:nosplit
//go:cgo_unsafe_args
func res_init() int32 {
	return libcCall(unsafe.Pointer(funcPC(res_init_trampoline)), nil)
}
func res_init_trampoline()

//go:linkname res_search net.res_search
//go:nosplit
//go:cgo_unsafe_args
func res_search(dname *byte, class int32, rtype int32, answer *byte, anslen int32) (int32, int32) {
	args := struct {
		dname                   *byte
		class, rtype            int32
		answer                  *byte
		anslen, retSize, retErr int32
	}{dname, class, rtype, answer, anslen, 0, 0}
	libcCall(unsafe.Pointer(funcPC(res_search_trampoline)), unsafe.Pointer(&args))
	return args.retSize, args.retErr
}
func res_search_trampoline()

//go:cgo_import_dynamic libc_res_search res_search "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_res_init res_init "/usr/lib/libSystem.B.dylib"
