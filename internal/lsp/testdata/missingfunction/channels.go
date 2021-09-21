package missingfunction

func channels(s string) {
	undefinedChannels(c()) //@suggestedfix("undefinedChannels", "quickfix")
}

func c() (<-chan string, chan string) {
	return make(<-chan string), make(chan string)
}
