package typeerrors

func x() { return nil } //@suggestedfix("nil", "quickfix", "")

func y() { return nil, "hello" } //@suggestedfix("nil", "quickfix", "")
