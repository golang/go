package main

// The Go testing framework requires this harness function to be placed in a separate file to properly track and record code coverage metrics
func harness(data []byte) {
	cmpLogTarget := []byte("FUZZING!")

	if len(data) < 4+len(cmpLogTarget) {
		return
	}

	if data[0] == 'F' {
		if data[1] == 'U' {
			if data[2] == 'Z' {
				if data[3] == 'Z' {
					if string(data[4:4+len(cmpLogTarget)]) == string(cmpLogTarget) {
						panic("FUZZFUZZING!")
					}
				}
			}
		}
	}
}
