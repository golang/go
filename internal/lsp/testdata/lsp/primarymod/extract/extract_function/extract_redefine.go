package extract

import "strconv"

func _() {
	i, err := strconv.Atoi("1")
	u, err := strconv.Atoi("2") //@extractfunc("u", ")")
	if i == u || err == nil {
		return
	}
}
