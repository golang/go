package other

type ImpP struct{} //@mark(OtherImpP, "ImpP")

func (*ImpP) Laugh() { //@mark(OtherLaughP, "Laugh")
}

type ImpS struct{} //@mark(OtherImpS, "ImpS")

func (ImpS) Laugh() { //@mark(OtherLaughS, "Laugh")
}

type ImpI interface { //@mark(OtherLaugher, "ImpI")
	Laugh() //@mark(OtherLaugh, "Laugh")
}

type Foo struct { //@implementations("Foo", Joker)
}

func (Foo) Joke() { //@mark(ImpJoker, "Joke"),implementations("Joke", Joke)
}

type CryType int

type Cryer interface { //@Cryer
	Cry(CryType) //@Cry,implementations("Cry", CryImpl)
}
