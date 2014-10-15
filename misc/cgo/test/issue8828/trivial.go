package issue8828

//void foo();
import "C"

func Bar() {
	C.foo()
}
