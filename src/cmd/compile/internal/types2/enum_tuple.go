package types2

import "strconv"

// enumTupleTagSentinel marks the synthetic tuple carrier struct used for enum
// tuple payloads. It is stored in Struct field tags for internal discrimination.
//
// This reduces the chance of accidentally treating a user-defined struct with
// fields named _0, _1, ... as a tuple payload.
const enumTupleTagSentinel = "__types2_enum_tuple__"

// isTuplePayloadStruct reports whether st is the synthetic tuple carrier struct
// used to model multi-arg enum variant payloads:
//
//	struct{ _0 T0; _1 T1; ... }
//
// Important: do NOT treat arbitrary user structs (e.g. type User struct{ Name string; ID int })
// as tuple payloads, even though they are structs, otherwise enum constructors/patterns
// would incorrectly "explode" them into multiple arguments/bindings.
func isTuplePayloadStruct(st *Struct) bool {
	if st == nil {
		return false
	}
	n := st.NumFields()
	if n == 0 {
		return false
	}
	for i := 0; i < n; i++ {
		f := st.Field(i)
		if f == nil {
			return false
		}
		// Require internal sentinel tag on every field to avoid collisions with user structs.
		if st.Tag(i) != enumTupleTagSentinel {
			return false
		}
		if f.name != "_"+strconv.Itoa(i) {
			return false
		}
	}
	return true
}
