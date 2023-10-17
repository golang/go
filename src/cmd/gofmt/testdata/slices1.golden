//gofmt -s

// Test cases for slice expression simplification.
package p

var (
	a [10]byte
	b [20]float32
	s []int
	t struct {
		s []byte
	}

	_ = a[0:]
	_ = a[1:10]
	_ = a[2:]
	_ = a[3:(len(a))]
	_ = a[len(a) : len(a)-1]
	_ = a[0:len(b)]
	_ = a[2:len(a):len(a)]

	_ = a[:]
	_ = a[:10]
	_ = a[:]
	_ = a[:(len(a))]
	_ = a[:len(a)-1]
	_ = a[:len(b)]
	_ = a[:len(a):len(a)]

	_ = s[0:]
	_ = s[1:10]
	_ = s[2:]
	_ = s[3:(len(s))]
	_ = s[len(a) : len(s)-1]
	_ = s[0:len(b)]
	_ = s[2:len(s):len(s)]

	_ = s[:]
	_ = s[:10]
	_ = s[:]
	_ = s[:(len(s))]
	_ = s[:len(s)-1]
	_ = s[:len(b)]
	_ = s[:len(s):len(s)]

	_ = t.s[0:]
	_ = t.s[1:10]
	_ = t.s[2:len(t.s)]
	_ = t.s[3:(len(t.s))]
	_ = t.s[len(a) : len(t.s)-1]
	_ = t.s[0:len(b)]
	_ = t.s[2:len(t.s):len(t.s)]

	_ = t.s[:]
	_ = t.s[:10]
	_ = t.s[:len(t.s)]
	_ = t.s[:(len(t.s))]
	_ = t.s[:len(t.s)-1]
	_ = t.s[:len(b)]
	_ = t.s[:len(t.s):len(t.s)]
)

func _() {
	s := s[0:]
	_ = s
}
