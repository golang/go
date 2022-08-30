package other

import "golang.org/lsptests/rename/crosspkg"

func Other() {
	crosspkg.Bar
	crosspkg.Foo() //@rename("Foo", "Flamingo")
}
