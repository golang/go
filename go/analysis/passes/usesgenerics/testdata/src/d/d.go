// want package:`features{typeSet}`

package d

type myInt int

func _() {
	// Sanity check that we can both detect local types and interfaces with
	// embedded defined types.
	type constraint interface {
		myInt
	}
}
