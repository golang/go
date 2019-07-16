package p1

type O map[string]map[string]string

func (opts O) RemoveOption(sect, opt string) bool {
	if _, ok := opts[sect]; !ok {
		return false
	}
	_, ok := opts[sect][opt]
	delete(opts[sect], opt)
	return ok
}
