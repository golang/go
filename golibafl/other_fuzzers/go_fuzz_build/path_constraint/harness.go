
package harness

func harness(data []byte) (bool) {
	cmpLogTarget := []byte("FUZZING!")

    if len(data) < 4+len(cmpLogTarget) {
        return false
    }

	if data[0] == 'F' {
		if data[1] == 'U' {
			if data[2] == 'Z' {
				if data[3] == 'Z' {
					if string(data[4:4+len(cmpLogTarget)]) == string(cmpLogTarget) {
						return true;
					}
				}
			}
		}
	}

	return false;
}
