package channel

func _() {
	var (
		aa = "123" //@item(channelAA, "aa", "string", "var")
		ab = 123   //@item(channelAB, "ab", "int", "var")
	)

	{
		type myChan chan int
		var mc myChan
		mc <- a //@complete(" //", channelAB, channelAA)
	}

	{
		var ac chan int //@item(channelAC, "ac", "chan int", "var")
		a <- a //@complete(" <-", channelAC, channelAA, channelAB)
	}

	{
		var foo chan int //@item(channelFoo, "foo", "chan int", "var")
		wantsInt := func(int) {} //@item(channelWantsInt, "wantsInt", "func(int)", "var")
		wantsInt(<-) //@rank(")", channelFoo, channelAB)
	}
}
