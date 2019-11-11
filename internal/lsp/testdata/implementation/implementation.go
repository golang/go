package implementation

type ImpP struct{} //@ImpP

func (*ImpP) Laugh() { //@mark(LaughP, "Laugh")
}

type ImpS struct{} //@ImpS

func (ImpS) Laugh() { //@mark(LaughS, "Laugh")
}

type ImpI interface { //@ImpI
	Laugh() //@mark(LaughI, "Laugh"),implementations("augh", LaughP),implementations("augh", OtherLaughP),implementations("augh", LaughS),implementations("augh", LaughL),implementations("augh", OtherLaughI),implementations("augh", OtherLaughS)
}

type Laugher interface { //@Laugher,implementations("augher", ImpP),implementations("augher", OtherImpP),implementations("augher", ImpI),implementations("augher", ImpS),implementations("augher", OtherImpI),implementations("augher", OtherImpS),
	Laugh() //@mark(LaughL, "Laugh"),implementations("augh", LaughP),implementations("augh", OtherLaughP),implementations("augh", LaughI),implementations("augh", LaughS),implementations("augh", OtherLaughI),implementations("augh", OtherLaughS)
}
