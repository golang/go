package exported_fields

// Used for testing exportedFields.
// Its exported fields are:
//   A1 [1]int
//   D bool
//   E int
//   F F
//   S *S
type (
	S struct {
		int
		*embed2
		embed
		E int // shadows embed.E
		alias
		A1
		*S
	}

	A1 [1]int

	embed struct {
		E string
	}

	embed2 struct {
		embed3
		F // shadows embed3.F
	}
	embed3 struct {
		F bool
	}
	alias = struct{ D bool }

	F int
)
