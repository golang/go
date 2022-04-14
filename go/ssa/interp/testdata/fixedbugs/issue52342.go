package main

func main() {
	var d byte

	d = 1
	d <<= 256
	if d != 0 {
		panic(d)
	}

	d = 1
	d >>= 256
	if d != 0 {
		panic(d)
	}
}
