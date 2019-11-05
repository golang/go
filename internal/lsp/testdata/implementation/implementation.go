package implementation

type ImpP struct{} //@ImpP

func (*ImpP) Laugh() { //@mark(LaughP, "Laugh")
}

type ImpS struct{} //@ImpS

func (ImpS) Laugh() { //@mark(LaughS, "Laugh")
}

type ImpI interface { //@ImpI
	Laugh() //@mark(LaughI, "Laugh"),implementations("augh", LaughP),implementations("augh", LaughS),implementations("augh", LaughL)
}

type Laugher interface { //@Laugher,implementations("augher", ImpP),implementations("augher", ImpI),implementations("augher", ImpS),
	Laugh() //@mark(LaughL, "Laugh"),implementations("augh", LaughP),implementations("augh", LaughI),implementations("augh", LaughS)
}
