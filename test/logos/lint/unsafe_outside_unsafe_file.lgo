// errorcheck -0 --pedantic

package p

import "unsafe" // ERROR "\[pedantic::unsafe_outside_unsafe_file\]"

var slot unsafe.Pointer

func store(p *int) {
	slot = unsafe.Pointer(p)
}
