// Tests of type asserts.
// Requires type parameters.
package typeassert

type fooer interface{ foo() string }

type X int

func (_ X) foo() string { return "x" }

func f[T fooer](x T) func() string {
	return x.foo
}

func main() {
	if f[X](0)() != "x" {
		panic("f[X]() != 'x'")
	}

	p := false
	func() {
		defer func() {
			if recover() != nil {
				p = true
			}
		}()
		f[fooer](nil) // panics on x.foo when T is an interface and nil.
	}()
	if !p {
		panic("f[fooer] did not panic")
	}
}
