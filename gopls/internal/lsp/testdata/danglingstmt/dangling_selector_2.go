package danglingstmt

import "golang.org/lsptests/foo"

func _() {
	foo. //@rank(" //", Foo)
	var _ = []string{foo.} //@rank("}", Foo)
}
