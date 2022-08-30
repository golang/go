package implementation

import "golang.org/lsptests/implementation/other"

type ImpP struct{} //@ImpP,implementations("ImpP", Laugher, OtherLaugher)

func (*ImpP) Laugh() { //@mark(LaughP, "Laugh"),implementations("Laugh", Laugh, OtherLaugh)
}

type ImpS struct{} //@ImpS,implementations("ImpS", Laugher, OtherLaugher)

func (ImpS) Laugh() { //@mark(LaughS, "Laugh"),implementations("Laugh", Laugh, OtherLaugh)
}

type Laugher interface { //@Laugher,implementations("Laugher", ImpP, OtherImpP, ImpS, OtherImpS)
	Laugh() //@Laugh,implementations("Laugh", LaughP, OtherLaughP, LaughS, OtherLaughS)
}

type Foo struct { //@implementations("Foo", Joker)
	other.Foo
}

type Joker interface { //@Joker
	Joke() //@Joke,implementations("Joke", ImpJoker)
}

type cryer int //@implementations("cryer", Cryer)

func (cryer) Cry(other.CryType) {} //@mark(CryImpl, "Cry"),implementations("Cry", Cry)

type Empty interface{} //@implementations("Empty")
