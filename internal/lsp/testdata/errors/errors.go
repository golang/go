package errors

import (
	"golang.org/lsptests/types"
)

func _() {
	bob.Bob() //@complete(".")
	types.b //@complete(" //", Bob_interface)
}
