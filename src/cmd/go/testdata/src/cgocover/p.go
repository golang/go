package p

/*
void
f(void)
{
}
*/
import "C"

var b bool

func F() {
	if b {
		for {
		}
	}
	C.f()
}
