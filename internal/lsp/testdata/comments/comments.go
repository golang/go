package comments

var p bool

//@complete(re"$")

func _() {
	var a int

	switch a {
	case 1:
		//@complete(re"$")
		_ = a
	}

	var b chan int
	select {
	case <-b:
		//@complete(re"$")
		_ = b
	}

	var (
		//@complete(re"$")
		_ = a
	)
}
