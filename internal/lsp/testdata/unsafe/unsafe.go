package unsafe

import (
	"unsafe"
)

// Pre-set this marker, as we don't have a "source" for it in this package.
/* unsafe.Sizeof */ //@item(Sizeof, "Sizeof", "invalid type", "text")

func _() {
	x := struct{}{}
	_ = unsafe.Sizeof(x) //@complete("z", Sizeof)
}
