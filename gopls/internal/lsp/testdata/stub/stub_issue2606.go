package stub

type I interface{ error }

type C int

var _ I = C(0) //@suggestedfix("C", "quickfix", "")
