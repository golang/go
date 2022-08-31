package c

import "golang.org/lsptests/rename/b"

func _() {
	b.Hello() //@rename("Hello", "Goodbye")
}
