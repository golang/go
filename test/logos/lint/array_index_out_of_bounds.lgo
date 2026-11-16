// errorcheck

package p

type token uint8

const (
	tokenLight token = iota
	tokenDark
	tokenCount
)

func (t token) String() string {
	names := [...]string{"light", "dark"}
	if t >= tokenCount {
		return "unknown"
	}
	return names[t]
}

func offByOne(t token) string {
	names := [...]string{"light", "dark"}
	if t > tokenCount {
		return "unknown"
	}
	return names[t] // ERROR "correctness::array_index_out_of_bounds"
}

func constantOnLeft(t token) string {
	names := [...]string{"light", "dark"}
	if tokenCount < t {
		return "unknown"
	}
	return names[t] // ERROR "correctness::array_index_out_of_bounds"
}

func inclusiveSafe(t token) string {
	names := [...]string{"light", "dark"}
	if t > tokenDark {
		return "unknown"
	}
	return names[t]
}

func largerBound(t token) string {
	names := [...]string{"light", "dark"}
	if t >= tokenCount+1 {
		return "unknown"
	}
	return names[t] // ERROR "correctness::array_index_out_of_bounds"
}

func signedSafe(i int8) string {
	names := [...]string{"light", "dark"}
	if i < 0 || i >= 2 {
		return "unknown"
	}
	return names[i]
}

func signedNegative(i int8) string {
	names := [...]string{"light", "dark"}
	if i >= 2 {
		return "unknown"
	}
	return names[i] // ERROR "correctness::array_index_out_of_bounds"
}

func fullWidth(t token) string {
	var names [256]string
	return names[t]
}

func unknownGuard(t token, valid bool) string {
	names := [...]string{"light", "dark"}
	if valid {
		return names[t]
	}
	return "unknown"
}

func changedIndex(t token) string {
	names := [...]string{"light", "dark"}
	t = tokenDark
	return names[t]
}
