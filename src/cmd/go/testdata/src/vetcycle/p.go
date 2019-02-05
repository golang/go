package p

type (
	_  interface{ m(B1) }
	A1 interface{ a(D1) }
	B1 interface{ A1 }
	C1 interface {
		B1 /* ERROR issue #18395 */
	}
	D1 interface{ C1 }
)

var _ A1 = C1 /* ERROR cannot use C1 */ (nil)
